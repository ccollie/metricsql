use lz4_flex::block::{
    compress_into, compress_prepend_size, decompress_size_prepended, get_maximum_output_size,
    CompressError, DecompressError,
};
use q_compress::{
    auto_compress, auto_decompress, data_types::NumberLike, errors::QCompressResult, Compressor,
    CompressorConfig, DEFAULT_COMPRESSION_LEVEL,
};

/// Returns the maximum output size of the compressed data.
/// Can be used to preallocate capacity on the output vector
#[inline]
pub fn lz4_get_maximum_output_size(input_len: usize) -> usize {
    get_maximum_output_size(input_len)
}

/// compress_lz4 appends compressed src to dst and returns
/// the appended dst.
pub fn compress_lz4(src: &[u8]) -> Vec<u8> {
    compress_prepend_size(src)
}

/// Compress all bytes of `input` into `output`.
/// output should be preallocated with a size of `lz4_get_maximum_output_size`.
///
/// Returns the number of bytes written (compressed) into `output`.
#[inline]
pub fn compress_lz4_into(input: &[u8], output: &mut [u8]) -> Result<usize, CompressError> {
    compress_into(input, output)
}

pub fn decompress_lz4(compressed: &[u8]) -> Result<Vec<u8>, DecompressError> {
    decompress_size_prepended(compressed)
}

pub fn decompress_lz4_into(dst: &mut Vec<u8>, compressed: &[u8]) -> Result<(), DecompressError> {
    let decompressed = decompress_size_prepended(compressed).unwrap();
    dst.extend_from_slice(&decompressed);
    Ok(())
}

pub fn uncompressed_size_lz4(input: &[u8]) -> Result<(usize, &[u8]), DecompressError> {
    let size = input.get(..4).ok_or(DecompressError::ExpectedAnotherByte)?;
    let size: [u8; 4] = size.try_into().unwrap();
    let uncompressed_size = u32::from_le_bytes(size) as usize;
    let rest = &input[4..];
    Ok((uncompressed_size, rest))
}

pub fn compress_quantile<T>(nums: &[T], config: CompressorConfig) -> Vec<u8>
where
    T: NumberLike,
{
    Compressor::<T>::from_config(config).simple_compress(nums)
}

pub fn compress_quantile_auto<T>(data: &[T]) -> Vec<u8>
where
    T: NumberLike,
{
    auto_compress(data, DEFAULT_COMPRESSION_LEVEL)
}

pub fn decompress_quantile_auto<T>(bytes: &[u8]) -> QCompressResult<Vec<T>>
where
    T: NumberLike,
{
    auto_decompress::<T>(bytes)
}

// var (
// compressCalls   = metrics.NewCounter(`vm_zstd_block_compress_calls_total`)
// decompressCalls = metrics.NewCounter(`vm_zstd_block_decompress_calls_total`)
//
// originalBytes   = metrics.NewCounter(`vm_zstd_block_original_bytes_total`)
// compressedBytes = metrics.NewCounter(`vm_zstd_block_compressed_bytes_total`)
// )
