static quantization_matrix: [i32; 64] = [
     8, 16, 19, 22, 26, 27, 29, 34,
    16, 16, 22, 24, 27, 29, 34, 37,
    19, 22, 26, 27, 29, 34, 34, 38,
    22, 22, 26, 27, 29, 34, 37, 40,
    22, 26, 27, 29, 32, 35, 40, 48,
    26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69,
    27, 29, 35, 38, 46, 56, 69, 83
];

const zig_zag_ordering: [i32; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
];

const dct_dc_coefficient_luminance_size_table: [(usize,usize); 9] = 
[
    (0b100,3),
    (0b00,2),
    (0b01,2),
    (0b101,3),
    (0b110,3),
    (0b1110,4),
    (0b11110,5),
    (0b111110,6),
    (0b1111110,7)
];

const dct_dc_coefficient_chrominance_size_table: [(usize,usize); 9] = 
[
    (0b00,2),
    (0b01,2),
    (0b10,2),
    (0b110,3),
    (0b1110,4),
    (0b11110,5),
    (0b111110,6),
    (0b1111110,7),
    (0b11111110,8)
];

#[derive(Default)]
struct macroblock
{
    headers:u8,
    header_size:usize,
    y1:u128,
    y1_size:usize,
    y2:u128,
    y2_size:usize,
    y3:u128,
    y3_size:usize,
    y4:u128,
    y4_size:usize,
    cb:u128,
    cb_size:usize,
    cr:u128,
    cr_size:usize
}


// https://www.ijera.com/papers/Vol5_issue5/Part%20-%201/P505018790.pdf
// Perform an approximate 8x8 2D DCT
// on a block
pub fn approximate_2d_dct(A: [i32;64]) -> [i32;64]
{
    
    // Output = (T*A*T(transposed))*(D^2)


    //T Matrix
    // 1 1 1  1  1  1 1  1
    // 0 1 0  0  0  0 −1 0
    // 1 0 0 −1 −1  0 0  1
    // 0 0 0 −1  1  0 0  0
    // 0 0 0  0  0  0 0  0
    // 0 0 0  0  0  0 0  0
    // 0 0 0  0  0  0 0  0
    // 0 0 0  0  0  0 0  0
    let mut  A_: [i32;64] = [0;64];
        for n in 0..=7
        {
            A_[0+n]=A[0+n]+A[8+n]+A[16+n]+A[24+n]+A[32+n]+A[40+n]+A[48+n]+A[56+n];
            A_[8+n]=A[8+n]-A[48+n];
            A_[16+n]=A[0+n]-A[24+n]-A[32+n]+A[56+n];
            A_[24+n]=0-A[24+n]+A[32+n];
            A_[32+n]=0;
            A_[40+n]=0;
            A_[48+n]=0;
            A_[56+n]=0;
        }
    

    let mut  B_: [i32;64] = [0;64];
    //T Transposed Matrix
    // 1  0  1  0  0  0  0  0
    // 1  1  0  0  0  0  0  0
    // 1  0  0  0  0  0  0  0
    // 1  0 -1 -1  0  0  0  0
    // 1  0 -1  1  0  0  0  0
    // 1  0  0  0  0  0  0  0
    // 1 -1  0  0  0  0  0  0
    // 1  0  1  0  0  0  0  0
        for n in 0..=7
        {
            B_[0+(8*n)]=A_[0+(n*8)]+A_[1+(8*n)]+A_[2+(8*n)]+A_[3+(8*n)]+A_[4+(8*n)]+A_[5+(8*n)]+A_[6+(8*n)]+A_[7+(8*n)];
            B_[1+(8*n)]=A_[1+(8*n)]-A_[6+(8*n)];
            B_[2+(8*n)]=A_[0+(8*n)]-A_[3+(8*n)]-A_[4+(8*n)]+A_[7+(8*n)];
            B_[3+(8*n)]=0-A_[3+(8*n)]+A_[4+(8*n)];
            B_[4+(8*n)]=0;
            B_[5+(8*n)]=0;
            B_[6+(8*n)]=0;
            B_[7+(8*n)]=0;
        
    }


    // D Matrix
    // 1/sqrt(8)    0     0     0         0   0   0   0
    // 0      1/sqrt(2)   0     0         0   0   0   0
    // 0            0    1/2    0         0   0   0   0
    // 0            0     0   1/sqrt(2)   0   0   0   0
    // 0            0     0     0         0   0   0   0
    // 0            0     0     0         0   0   0   0
    // 0            0     0     0         0   0   0   0
    // 0            0     0     0         0   0   0   0

    // D ^2 Matrix
    // 1/8    0     0     0      0   0   0   0
    // 0     1/2    0     0      0   0   0   0
    // 0      0    1/4    0      0   0   0   0
    // 0      0     0    1/2     0   0   0   0
    // 0      0     0     0      0   0   0   0
    // 0      0     0     0      0   0   0   0
    // 0      0     0     0      0   0   0   0
    // 0      0     0     0      0   0   0   0
    //
    //1/8 is >> 3
    //1/2 is >> 1
    //1/4 is >> 2
    
    let mut  B: [i32;64] = [0;64];
    
        for n in 0..=7
        {
            B[0+(n)]=B_[0+(n)]>>3;
            B[8+(n)]=B_[8+(n)]>>1;
            B[16+(n)]=B_[16+(n)]>>2;
            B[24+(n)]=B_[24+(n)]>>1;
            //Rest are zeroes
        
    }
    B
}

//Perform quantization on a DCT transformed block
pub fn quantize_block(A: [i32;64], quant_matrix: [i32;64]) -> [i32;64]
{

    let mut  A_: [i32;64] = [0;64];

    //AC & DC Coefficient being Quantized. Quantizer_Scale is 1 for this encoder.
    for i in 0..=63
    {
        //Divide by quant matrix
        A_[i] = (A[i])/quant_matrix[i]
        
    }
    A_
}

//Perform zig-zag encoding on quantized block.
//Only include first 4 AC co-efficients in a block.
pub fn zigzag_and_block_bitstream_encoding(A: [i32;64], prev_dc_coeff: &mut i32, dc_size_table: [(usize,usize); 9] ) -> (u128,usize) //(encoding, size)
{
    let mut result: u128 = 0;
    let mut result_size: usize = 0;

    //Encode DC difference Coefficient  
    let mut dc_coefficient_delta = A[0]-*prev_dc_coeff;
    
    *prev_dc_coeff=A[0];

    let mut size: usize = 8;
    let mut is_msb_1 = true;
    for i in 8..0
    {
        is_msb_1 = ((dc_coefficient_delta>>(size-1))==1);
        if(is_msb_1)
        {
            break;
        }
        size-=1;
    }

    result=result&(dc_size_table[size].0) as u128;
    result_size = dc_size_table[size].1;
    if(dc_coefficient_delta<0) //Subtract 1 from coefficient size if negative
    {
        result=result-1;
    }
    result=result<<size;

    result=result&dc_coefficient_delta as u128;
    result_size+=size;

    let mut coefficient = A[1];
    let mut run_length: u128 = 0;
    if(coefficient!=0)
    {
    for i in 1..=63
    {
        if(A[(zig_zag_ordering[i] as usize)]==0)
        {
            run_length=run_length+1;
        }
        else {
            //Record VLC encoding the easy way
            //Avoid the Huffman table for this draft
            //Escape Code
            result=result<<6;
            result=result&0b000001;
            result_size+=6;

            //Length (6 bit code)
            result=result <<6;
            result = result & run_length;
            result_size+=6;

            //Level (8 bit code)
            result=result <<8;
            result = result & coefficient as u128;
            result_size+=8;

            run_length=0;
            coefficient = A[(zig_zag_ordering[i] as usize)];
        }

    }
}
        //Record End of Block
        result = result << 2;
        result = result & 0b10;
        result_size+=2;

    (result, result_size)
}

pub fn block_encode(A: [i32;64], previous_dc:&mut i32, dc_size_table: [(usize,usize); 9], quant_matrix: [i32;64]) -> (u128,usize)
{
    zigzag_and_block_bitstream_encoding(quantize_block(approximate_2d_dct(A),quant_matrix),previous_dc, dc_size_table)
}


pub fn encode_macro_block(Y1: [i32;64], Y2: [i32;64], Y3: [i32;64], Y4: [i32;64], Cb: [i32;64], Cr: [i32;64],  previous_y_dc: &mut i32, previous_cb_dc: &mut i32, previous_cr_dc: &mut i32, luminance_dc_size_table: [(usize,usize); 9], chrominance_dc_size_table: [(usize,usize); 9], quant_matrix: [i32;64]) -> macroblock
{

    let mut m_block: macroblock = macroblock::default();
    //Bit 2 - Address Increment
    //Bits [1:0] - IQ macroblock for Y1,Y2,Y3,Y4,Cb and Cr
    m_block.headers=0b101;
    m_block.header_size=3;
    (m_block.y1, m_block.y1_size) = block_encode(Y1,previous_y_dc,luminance_dc_size_table, quant_matrix);
    (m_block.y2, m_block.y2_size) = block_encode(Y2,previous_y_dc,luminance_dc_size_table, quant_matrix);
    (m_block.y3, m_block.y3_size) = block_encode(Y3,previous_y_dc,luminance_dc_size_table, quant_matrix);
    (m_block.y4, m_block.y4_size) = block_encode(Y4,previous_y_dc,luminance_dc_size_table, quant_matrix);
    (m_block.cb, m_block.cb_size) = block_encode(Cb,previous_cb_dc,chrominance_dc_size_table, quant_matrix);
    (m_block.cr, m_block.cr_size) = block_encode(Cr,previous_cr_dc,chrominance_dc_size_table, quant_matrix);
    return m_block;
}