//! A spike: a VCF chunk parser, and a kinship product and an eigendecomposition
//! through faer, to measure Rust against the Python and numpy pynei has.
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

use faer::{Mat, MatRef, Side};
use flate2::read::MultiGzDecoder;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(not(target_family = "wasm"))]
use rayon::prelude::*;

const MISSING: i8 = -1;

fn open(path: &str) -> PyResult<Box<dyn BufRead + Send>> {
    let file = File::open(path).map_err(|e| PyValueError::new_err(format!("{path}: {e}")))?;
    let reader: Box<dyn Read + Send> = if path.ends_with(".gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::with_capacity(4 << 20, reader)))
}

struct ParsedVar {
    chrom: String,
    pos: u64,
    gts: Vec<i8>,
}

fn parse_allele(bytes: &[u8]) -> i8 {
    if bytes.is_empty() || bytes[0] == b'.' {
        return MISSING;
    }
    let mut allele: i8 = 0;
    for &b in bytes {
        allele = allele * 10 + (b - b'0') as i8;
    }
    allele
}

/// The genotype of one sample, its alleles into `out`, missing ones as MISSING.
fn parse_gt(field: &[u8], gt_idx: usize, ploidy: usize, out: &mut [i8]) {
    // the GT is the gt_idx-th colon separated value of the field
    let mut value = field;
    for _ in 0..gt_idx {
        match memchr::memchr(b':', value) {
            Some(i) => value = &value[i + 1..],
            None => {
                value = b".";
                break;
            }
        }
    }
    if let Some(i) = memchr::memchr(b':', value) {
        value = &value[..i];
    }
    let mut allele_idx = 0;
    for allele in value.split(|&b| b == b'/' || b == b'|') {
        if allele_idx >= ploidy {
            break;
        }
        out[allele_idx] = parse_allele(allele);
        allele_idx += 1;
    }
    for slot in out.iter_mut().skip(allele_idx) {
        *slot = MISSING;
    }
}

fn parse_line(line: &[u8], num_samples: usize, ploidy: usize) -> Result<ParsedVar, String> {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let mut fields = line.split(|&b| b == b'\t');
    let chrom = fields.next().ok_or("no chrom")?;
    let pos = fields.next().ok_or("no pos")?;
    let pos: u64 = std::str::from_utf8(pos)
        .ok()
        .and_then(|p| p.parse().ok())
        .ok_or("bad pos")?;
    // id, ref, alt, qual, filter, info
    for _ in 0..6 {
        fields.next().ok_or("short line")?;
    }
    let format = fields.next().ok_or("no format")?;
    let gt_idx = format
        .split(|&b| b == b':')
        .position(|key| key == b"GT")
        .ok_or("no GT in format")?;
    let mut gts = vec![MISSING; num_samples * ploidy];
    let mut num_seen = 0;
    for (sample_idx, field) in fields.enumerate() {
        if sample_idx >= num_samples {
            return Err("more samples than in the header".into());
        }
        parse_gt(field, gt_idx, ploidy, &mut gts[sample_idx * ploidy..(sample_idx + 1) * ploidy]);
        num_seen += 1;
    }
    if num_seen != num_samples {
        return Err(format!("{num_seen} samples in a line, {num_samples} in the header"));
    }
    Ok(ParsedVar {
        chrom: String::from_utf8_lossy(chrom).into_owned(),
        pos,
        gts,
    })
}

fn guess_ploidy(line: &[u8]) -> usize {
    // the first sample field of the first variant, its GT, counting alleles
    let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
    if fields.len() < 10 {
        return 2;
    }
    let mut gt = fields[9];
    if let Some(i) = memchr::memchr(b':', gt) {
        gt = &gt[..i];
    }
    gt.split(|&b| b == b'/' || b == b'|').count().max(1)
}

struct Chunk {
    gts: Array3<i8>,
    chroms: Vec<String>,
    poss: Vec<u64>,
}

fn parse_vcf_chunks(path: &str, num_vars_per_chunk: usize) -> PyResult<(Vec<String>, Vec<Chunk>)> {
    let mut reader = open(path)?;
    let mut line = Vec::new();
    let samples: Vec<String>;
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line).map_err(|e| PyValueError::new_err(e.to_string()))? == 0 {
            return Err(PyValueError::new_err("no header line"));
        }
        if line.starts_with(b"##") {
            continue;
        }
        if line.starts_with(b"#CHROM") {
            let text = String::from_utf8_lossy(&line);
            samples = text.trim_end().split('\t').skip(9).map(|s| s.to_string()).collect();
            break;
        }
        return Err(PyValueError::new_err("a line before the header that is not metadata"));
    }
    let num_samples = samples.len();
    let mut chunks = Vec::new();
    let mut ploidy = 0;
    let mut lines: Vec<Vec<u8>> = Vec::with_capacity(num_vars_per_chunk);
    loop {
        line.clear();
        let done = reader.read_until(b'\n', &mut line).map_err(|e| PyValueError::new_err(e.to_string()))? == 0;
        if !done && line.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        if !done {
            if ploidy == 0 {
                ploidy = guess_ploidy(&line);
            }
            lines.push(std::mem::take(&mut line));
        }
        if lines.len() == num_vars_per_chunk || (done && !lines.is_empty()) {
            // wasm has no threads: there rayon is not linked in and the parse is serial
            #[cfg(not(target_family = "wasm"))]
            let parsed: Result<Vec<ParsedVar>, String> = lines
                .par_iter()
                .map(|l| parse_line(l, num_samples, ploidy))
                .collect();
            #[cfg(target_family = "wasm")]
            let parsed: Result<Vec<ParsedVar>, String> = lines
                .iter()
                .map(|l| parse_line(l, num_samples, ploidy))
                .collect();
            let parsed = parsed.map_err(PyValueError::new_err)?;
            let num_vars = parsed.len();
            let mut gts = Array3::<i8>::zeros((num_vars, num_samples, ploidy));
            let flat = gts.as_slice_mut().unwrap();
            let row_len = num_samples * ploidy;
            #[cfg(not(target_family = "wasm"))]
            flat.par_chunks_mut(row_len).zip(parsed.par_iter()).for_each(|(row, var)| {
                row.copy_from_slice(&var.gts);
            });
            #[cfg(target_family = "wasm")]
            flat.chunks_mut(row_len).zip(parsed.iter()).for_each(|(row, var)| {
                row.copy_from_slice(&var.gts);
            });
            chunks.push(Chunk {
                gts,
                chroms: parsed.iter().map(|v| v.chrom.clone()).collect(),
                poss: parsed.iter().map(|v| v.pos).collect(),
            });
            lines.clear();
        }
        if done {
            break;
        }
    }
    Ok((samples, chunks))
}

/// A chunk as python sees it: the genotypes, the chromosomes and the positions.
type PyChunk<'py> = (Bound<'py, PyArray3<i8>>, Vec<String>, Vec<u64>);

/// It parses a VCF into chunks: the samples and a list of (gts, chroms, poss).
#[pyfunction]
#[pyo3(signature = (path, num_vars_per_chunk = 5000))]
fn parse_vcf<'py>(
    py: Python<'py>,
    path: &str,
    num_vars_per_chunk: usize,
) -> PyResult<(Vec<String>, Vec<PyChunk<'py>>)> {
    let (samples, chunks) = py.detach(|| parse_vcf_chunks(path, num_vars_per_chunk))?;
    let out = chunks
        .into_iter()
        .map(|c| (c.gts.into_pyarray(py), c.chroms, c.poss))
        .collect();
    Ok((samples, out))
}

fn mat_ref<'a>(array: &'a PyReadonlyArray2<'_, f64>) -> PyResult<MatRef<'a, f64>> {
    let shape = array.shape();
    let slice = array
        .as_slice()
        .map_err(|_| PyValueError::new_err("the array must be C contiguous"))?;
    Ok(MatRef::from_row_major_slice(slice, shape[0], shape[1]))
}

fn to_array(mat: MatRef<'_, f64>) -> Array2<f64> {
    let (nrows, ncols) = (mat.nrows(), mat.ncols());
    Array2::from_shape_fn((nrows, ncols), |(i, j)| mat[(i, j)])
}

/// z'z for z with one row per variant and one column per sample.
#[pyfunction]
fn zz<'py>(py: Python<'py>, z: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let z = mat_ref(&z)?;
    let result: Mat<f64> = py.detach(|| z.transpose() * z);
    Ok(to_array(result.as_ref()).into_pyarray(py))
}

/// An eigendecomposition as python sees it: the eigenvalues and the eigenvectors.
type PyEigen<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);

/// The eigenvalues and eigenvectors of a symmetric matrix, ascending, as numpy.linalg.eigh.
#[pyfunction]
fn eigh<'py>(
    py: Python<'py>,
    k: PyReadonlyArray2<'py, f64>,
) -> PyResult<PyEigen<'py>> {
    let k = mat_ref(&k)?;
    let (vals, vecs) = py.detach(|| {
        let eig = k.self_adjoint_eigen(Side::Lower).expect("eigendecomposition failed");
        let s = eig.S();
        let u = eig.U();
        let vals: Vec<f64> = (0..s.dim()).map(|i| s[i]).collect();
        (vals, to_array(u))
    });
    Ok((Array1::from(vals).into_pyarray(py), vecs.into_pyarray(py)))
}

#[pyfunction]
fn set_num_threads(num_threads: usize) {
    #[cfg(not(target_family = "wasm"))]
    faer::set_global_parallelism(if num_threads <= 1 {
        faer::Par::Seq
    } else {
        faer::Par::rayon(num_threads)
    });
    // wasm has no threads, so faer is always sequential there
    #[cfg(target_family = "wasm")]
    {
        let _ = num_threads;
        faer::set_global_parallelism(faer::Par::Seq);
    }
}

#[pymodule]
fn pynei_spike(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_vcf, m)?)?;
    m.add_function(wrap_pyfunction!(zz, m)?)?;
    m.add_function(wrap_pyfunction!(eigh, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    Ok(())
}
