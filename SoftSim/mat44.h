#pragma once

// stores column vectors in column major order
template <typename T>
class XMatrix44
{
public:

	__forceinline XMatrix44() { memset(columns, 0, sizeof(columns)); }
	__forceinline XMatrix44(const T* d) { BHAssert(d); memcpy(columns, d, sizeof(*this)); }
	__forceinline XMatrix44(T c11, T c21, T c31, T c41,
				 T c12, T c22, T c32, T c42,
			    T c13, T c23, T c33, T c43,
				 T c14, T c24, T c34, T c44)
	{
		columns[0][0] = c11;
		columns[0][1] = c21;
		columns[0][2] = c31;
		columns[0][3] = c41;

		columns[1][0] = c12;
		columns[1][1] = c22;
		columns[1][2] = c32;
		columns[1][3] = c42;

		columns[2][0] = c13;
		columns[2][1] = c23;
		columns[2][2] = c33;
		columns[2][3] = c43;

		columns[3][0] = c14;
		columns[3][1] = c24;
		columns[3][2] = c34;
		columns[3][3] = c44;
	}

	__forceinline operator T* () { return &columns[0][0]; }
	__forceinline operator const T* () const { return &columns[0][0]; }

	__forceinline void operator =  ( const XMatrix44<T>& sSource ) {
		memcpy( this, &sSource, sizeof( *this ) );
	}

	__forceinline void CopyFrom ( const XMatrix44<T>& sSource ) {
		memcpy( this, &sSource, sizeof( *this ) );
	}

	// right multiply
	__forceinline XMatrix44<T> operator * (const XMatrix44<T>& rhs) const
	{
		XMatrix44<T> r;
		MatrixMultiply(*this, rhs, r);
		return r;
	}

	// right multiply
	__forceinline XMatrix44<T>& operator *= (const XMatrix44<T>& rhs)
	{
		XMatrix44<T> r;
		MatrixMultiply(*this, rhs, r);
		*this = r;

		return *this;
	}

	// scalar multiplication
	__forceinline XMatrix44<T>& operator *= (const T& s)
	{
		for (int c=0; c < 4; ++c)
		{
			for (int r=0; r < 4; ++r)
			{
				columns[c][r] *= s;
			}
		}

		return *this;
	}

	__forceinline void MatrixMultiply(const T* __restrict lhs, const T* __restrict rhs, T* __restrict result) const {
		BHAssert( lhs != rhs);
		BHAssert( lhs != result);
		BHAssert( rhs != result);
		
		for (int i=0; i < 4; ++i) {
			for (int j=0; j < 4; ++j) {
				result[j*4+i]  = rhs[j*4+0]*lhs[i+0]; 
				result[j*4+i] += rhs[j*4+1]*lhs[i+4];
				result[j*4+i] += rhs[j*4+2]*lhs[i+8];
				result[j*4+i] += rhs[j*4+3]*lhs[i+12];
			}
		}
	}

	static __forceinline XMatrix44 Mul( const XMatrix44& M1, const XMatrix44& M2 ) {
		// NOTE: Copy form XMMatrixMultiply in DirectXMathMatrix.inl

		Matrix44 mResult;
		// Use vW to hold the original row
		XMVECTOR vW = M1.r[ 0 ];
		// Splat the component X,Y,Z then W
		XMVECTOR vX = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 0, 0, 0, 0 ) );
		XMVECTOR vY = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 1, 1, 1, 1 ) );
		XMVECTOR vZ = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 2, 2, 2, 2 ) );
		vW = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 3, 3, 3, 3 ) );
		// Perform the operation on the first row
		vX = _mm_mul_ps( vX, M2.r[ 0 ] );
		vY = _mm_mul_ps( vY, M2.r[ 1 ] );
		vZ = _mm_mul_ps( vZ, M2.r[ 2 ] );
		vW = _mm_mul_ps( vW, M2.r[ 3 ] );
		// Perform a binary add to reduce cumulative errors
		vX = _mm_add_ps( vX, vZ );
		vY = _mm_add_ps( vY, vW );
		vX = _mm_add_ps( vX, vY );
		mResult.r[ 0 ] = vX;
		// Repeat for the other 3 rows
		vW = M1.r[ 1 ];
		vX = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 0, 0, 0, 0 ) );
		vY = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 1, 1, 1, 1 ) );
		vZ = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 2, 2, 2, 2 ) );
		vW = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 3, 3, 3, 3 ) );
		vX = _mm_mul_ps( vX, M2.r[ 0 ] );
		vY = _mm_mul_ps( vY, M2.r[ 1 ] );
		vZ = _mm_mul_ps( vZ, M2.r[ 2 ] );
		vW = _mm_mul_ps( vW, M2.r[ 3 ] );
		vX = _mm_add_ps( vX, vZ );
		vY = _mm_add_ps( vY, vW );
		vX = _mm_add_ps( vX, vY );
		mResult.r[ 1 ] = vX;
		vW = M1.r[ 2 ];
		vX = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 0, 0, 0, 0 ) );
		vY = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 1, 1, 1, 1 ) );
		vZ = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 2, 2, 2, 2 ) );
		vW = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 3, 3, 3, 3 ) );
		vX = _mm_mul_ps( vX, M2.r[ 0 ] );
		vY = _mm_mul_ps( vY, M2.r[ 1 ] );
		vZ = _mm_mul_ps( vZ, M2.r[ 2 ] );
		vW = _mm_mul_ps( vW, M2.r[ 3 ] );
		vX = _mm_add_ps( vX, vZ );
		vY = _mm_add_ps( vY, vW );
		vX = _mm_add_ps( vX, vY );
		mResult.r[ 2 ] = vX;
		vW = M1.r[ 3 ];
		vX = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 0, 0, 0, 0 ) );
		vY = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 1, 1, 1, 1 ) );
		vZ = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 2, 2, 2, 2 ) );
		vW = XM_PERMUTE_PS( vW, _MM_SHUFFLE( 3, 3, 3, 3 ) );
		vX = _mm_mul_ps( vX, M2.r[ 0 ] );
		vY = _mm_mul_ps( vY, M2.r[ 1 ] );
		vZ = _mm_mul_ps( vZ, M2.r[ 2 ] );
		vW = _mm_mul_ps( vW, M2.r[ 3 ] );
		vX = _mm_add_ps( vX, vZ );
		vY = _mm_add_ps( vY, vW );
		vX = _mm_add_ps( vX, vY );
		mResult.r[ 3 ] = vX;

		return mResult;
	}

	__forceinline void MulBy( const XMatrix44<T>& M ) {
		*this = Mul( *this, M );
	}

	
	float columns[4][4];

	static XMatrix44<T> kIdentity;
};



template<typename T>
__forceinline XMatrix44<T> Transpose(const XMatrix44<T>& m)
{
	XMatrix44<float> inv;

	// transpose
	for (BHUint32 c=0; c < 4; ++c)
	{
		for (BHUint32 r=0; r < 4; ++r)
		{
			inv.columns[c][r] = m.columns[r][c];
		}
	}

	return inv;
}

template <typename T>
__forceinline XMatrix44<T> AffineInverse(const XMatrix44<T>& m)
{
	XMatrix44<T> inv;
	
	// transpose upper 3x3
	for (int c=0; c < 3; ++c)
	{
		for (int r=0; r < 3; ++r)
		{
			inv.columns[c][r] = m.columns[r][c];
		}
	}
	
	// multiply -translation by upper 3x3 transpose
	inv.columns[3][0] = -m.columns[3][0] * m.columns[0][0] - m.columns[3][1] * m.columns[0][1] - m.columns[3][2] * m.columns[0][2];
	inv.columns[3][1] = -m.columns[3][0] * m.columns[1][0] - m.columns[3][1] * m.columns[1][1] - m.columns[3][2] * m.columns[1][2];
	inv.columns[3][2] = -m.columns[3][0] * m.columns[2][0] - m.columns[3][1] * m.columns[2][1] - m.columns[3][2] * m.columns[2][2];
	inv.columns[3][3] = 1.0f;

	return inv;	
}

// convenience
typedef XMatrix44<float> Mat44;
typedef XMatrix44<float> Matrix44;


