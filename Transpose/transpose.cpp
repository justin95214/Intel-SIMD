

#include <iostream>
#include <iomanip>
#include <intrin.h>

using namespace std;

void PrintMat(__m128i mat[8]) {
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++)
			cout << setw(2) << mat[i].m128i_i16[j] << ' ';
		cout << endl;
	}
	cout << endl;
}

void TranspSIMD(short in[8][8]) {

	__m128i a[8];
	__m128i b[8];
	__m128i c[8];
	__m128i d[8];

	for (size_t i = 0; i < 8; i++)
	{
		a[i] = _mm_set_epi16(in[i][7], in[i][6], in[i][5], in[i][4], in[i][3], in[i][2], in[i][1], in[i][0]);
	}
	
	PrintMat(a);
	printf("\n");


	for (size_t i = 0; i < 8; i=i+2)
	{
	b[i] = _mm_unpacklo_epi16(a[i],a[i+1]);
	b[i+1] = _mm_unpackhi_epi16(a[i],a[i+1]);
	}
	PrintMat(b);
	printf("\n");

		c[0] = _mm_unpacklo_epi32(b[0], b[2]);
		c[1] = _mm_unpackhi_epi32(b[0], b[2]);
		c[2] = _mm_unpacklo_epi32(b[1], b[3]);
		c[3] = _mm_unpackhi_epi32(b[1], b[3]);
		c[4] = _mm_unpacklo_epi32(b[4], b[6]);
		c[5] = _mm_unpackhi_epi32(b[4], b[6]);
		c[6] = _mm_unpacklo_epi32(b[5], b[7]);
		c[7] = _mm_unpackhi_epi32(b[5], b[7]);
		
	PrintMat(c);
	printf("\n");


	for (size_t i = 0; i < 8; i = i + 2)
	{
		d[i]   = _mm_unpacklo_epi64(c[i/2], c[i/2 + 4]);
		d[i+1] = _mm_unpackhi_epi64(c[i/2], c[i/2 + 4]);
	}
	PrintMat(d);
}


int main() {
	short mat[8][8];
	int num = 1;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++, num += 1.0f) {
			mat[i][j] = num;

		}

	}
	
	TranspSIMD(mat);

	return 0;

}