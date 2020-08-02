//-------------------matrix.hpp------------------------------------------------//
#include <cstddef>
#include <ostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <string>
#include <cassert>
#include <utility>
#include <time.h>

using namespace std;

struct MatrixIndex {
	std::size_t Row, Column;
};

class Matrix {
private:
	std::vector<double> m_Data;
	std::size_t m_Row = 0, m_Column = 0;

public:
	Matrix() noexcept = default;
	Matrix(std::size_t row, std::size_t column);
	Matrix(double value, std::size_t row, std::size_t column);
	Matrix(std::vector<double> data, std::size_t row, std::size_t column);
	Matrix(const Matrix& matrix);
	Matrix(Matrix&& matrix) noexcept;
	~Matrix() = default;

public:
	Matrix& operator=(const Matrix& matrix);
	Matrix& operator=(Matrix&& matrix) noexcept;
	double operator[](const MatrixIndex& index) const noexcept;
	double& operator[](const MatrixIndex& index) noexcept;
	double operator()(std::size_t row, std::size_t column) const noexcept;
	double& operator()(std::size_t row, std::size_t column) noexcept;
	Matrix operator+(const Matrix& matrix) const;
	Matrix operator-(const Matrix& matrix) const;
	Matrix operator*(const float value);

	Matrix& operator+=(const Matrix& matrix) noexcept;
	Matrix& operator-=(const Matrix& matrix) noexcept;
	Matrix& operator*=(double value) noexcept;


public:
	void Clear() noexcept;
	bool IsEmpty() const noexcept;
	void Swap(Matrix& matrix) noexcept;
	std::size_t GetRow() const noexcept;
	std::size_t GetColumn() const noexcept;

	void Transpose();
	void HadamardProduct(const Matrix& matrix) noexcept;
};

Matrix operator*(double value, const Matrix& matrix);
std::ostream& operator<<(std::ostream& stream, const Matrix& matrix);


//-------------------activation .hpp------------------------------------------------//

class ActivationFunction {
private:
	Matrix(*m_Function)(const Matrix&);
	Matrix(*m_Derivative)(const Matrix&);

public:
	ActivationFunction(Matrix(*function)(const Matrix&), Matrix(*derivative)(const Matrix&)) noexcept;
	ActivationFunction(const ActivationFunction& function) noexcept = default;
	~ActivationFunction() = default;

public:
	ActivationFunction& operator=(const ActivationFunction& function) noexcept = default;
	Matrix operator()(const Matrix& x) const;

public:
	Matrix GetGradient(const Matrix& x) const;
};

extern const ActivationFunction Sigmoid;
extern const ActivationFunction ReLU;


//-------------------neuron.hpp------------------------------------------------//


class WeightedNeuron {
public:
	Matrix m_X;
	Matrix m_W;
	Matrix m_U;
	Matrix m_DW;
	Matrix m_DB;
	double m_B;

public:
	explicit WeightedNeuron(std::size_t inputSize);
	WeightedNeuron(const WeightedNeuron& neuron) = delete;
	~WeightedNeuron() = default;

public:
	WeightedNeuron& operator=(const WeightedNeuron& neuron) = delete;

public:
	Matrix Forward(Matrix& x);
	Matrix Backward(Matrix& d);
	void Update(double alpha);
};

class ActivationNeuron {
private:
	Matrix m_U;
	Matrix m_Y;
	ActivationFunction m_F;

public:
	explicit ActivationNeuron(ActivationFunction function);
	ActivationNeuron(const ActivationNeuron& neuron) = delete;
	~ActivationNeuron() = default;

public:
	ActivationNeuron& operator=(const ActivationNeuron& neuron) = delete;

public:
	Matrix Forward(const Matrix& x);
	Matrix Backward(const Matrix& d);
};



//---------loss function.hpp------------------------------------------------------//

class LossFunction {
private:
	double(*m_Function)(const Matrix&, const Matrix&);
	Matrix(*m_Derivative)(const Matrix&, const Matrix&);

public:
	LossFunction(double(*function)(const Matrix&, const Matrix&), Matrix(*derivative)(const Matrix&, const Matrix&)) noexcept;
	LossFunction(const LossFunction& function) noexcept = default;
	~LossFunction() = default;

public:
	LossFunction& operator=(const LossFunction& function) noexcept = default;
	double operator()(const Matrix& y, const Matrix& a) const;

public:
	Matrix GetGradient(const Matrix& y, const Matrix& a) const;
};

extern const LossFunction MSE;


//--------------------------cpp------------------------------------------//
//--------------------------matrix------------------------------------------//


Matrix::Matrix(std::size_t row, std::size_t column)
	: m_Data(row * column), m_Row(row), m_Column(column) {
	assert((row == 0 && column == 0) || (row > 0 && column > 0));
}
Matrix::Matrix(double value, std::size_t row, std::size_t column)
	: m_Data(row * column, value), m_Row(row), m_Column(column) {
	assert((row == 0 && column == 0) || (row > 0 && column > 0));
}
Matrix::Matrix(std::vector<double> data, std::size_t row, std::size_t column)
	: m_Data(data), m_Row(row), m_Column(column) {
	assert((row == 0 && column == 0) || (row > 0 && column > 0));
	assert(row * column == m_Data.size());
}
Matrix::Matrix(const Matrix& matrix)
	: m_Data(matrix.m_Data), m_Row(matrix.m_Row), m_Column(matrix.m_Column) {
}
Matrix::Matrix(Matrix&& matrix) noexcept
	: m_Data(std::move(matrix.m_Data)), m_Row(matrix.m_Row), m_Column(matrix.m_Column) {
	matrix.m_Row = matrix.m_Column = 0;
}

Matrix& Matrix::operator=(const Matrix& matrix) {
	m_Data = matrix.m_Data;
	m_Row = matrix.m_Row;
	m_Column = matrix.m_Column;

	return *this;
}
Matrix& Matrix::operator=(Matrix&& matrix) noexcept {
	m_Data = std::move(matrix.m_Data);
	m_Row = matrix.m_Row;
	m_Column = matrix.m_Column;

	matrix.m_Row = matrix.m_Column = 0;

	return *this;
}
double Matrix::operator[](const MatrixIndex& index) const noexcept {
	return (*this)(index.Row, index.Column);
}
double& Matrix::operator[](const MatrixIndex& index) noexcept {
	return (*this)(index.Row, index.Column);
}
double Matrix::operator()(std::size_t row, std::size_t column) const noexcept {
	assert(row < m_Row);
	assert(column < m_Column);

	return m_Data[row * m_Column + column];
}
double& Matrix::operator()(std::size_t row, std::size_t column) noexcept {
	assert(row < m_Row);
	assert(column < m_Column);

	return m_Data[row * m_Column + column];
}
Matrix Matrix::operator+(const Matrix& matrix) const {
	return Matrix(*this) += matrix;
}
Matrix Matrix::operator-(const Matrix& matrix) const {
	return Matrix(*this) -= matrix;
}
Matrix Matrix::operator*(const float value) {

	float matrix1[1][4] = { (*this)(0,0),(*this)(0,1),(*this)(0,2),(*this)(0,3) };
	__m128 b = _mm_loadu_ps(&matrix1[0][0]);
	__m128 c = _mm_load1_ps(&value);
	__m128 mul = _mm_mul_ps(b, c);
	Matrix out({ mul.m128_f32[0] , mul.m128_f32[1],mul.m128_f32[2] , mul.m128_f32[3] }, 1, 4);
	return out;
}


Matrix& Matrix::operator+=(const Matrix& matrix) noexcept {
	assert(m_Row == matrix.m_Row);
	assert(m_Column == matrix.m_Column);

	for (std::size_t i = 0; i < m_Data.size(); ++i) {
		m_Data[i] += matrix.m_Data[i];
	}

	return *this;
}
Matrix& Matrix::operator-=(const Matrix& matrix) noexcept {
	assert(m_Row == matrix.m_Row);
	assert(m_Column == matrix.m_Column);

	for (std::size_t i = 0; i < m_Data.size(); ++i) {
		m_Data[i] -= matrix.m_Data[i];
	}

	return *this;
}
Matrix& Matrix::operator*=(double value) noexcept {
	for (std::size_t i = 0; i < m_Data.size(); ++i) {
		m_Data[i] *= value;

	}

	return *this;
}


void Matrix::Clear() noexcept {
	m_Data.clear();
	m_Row = m_Column = 0;
}
bool Matrix::IsEmpty() const noexcept {
	return m_Data.empty();
}
void Matrix::Swap(Matrix& matrix) noexcept {
	if (this == &matrix) return;

	std::swap(m_Data, matrix.m_Data);
	std::swap(m_Row, matrix.m_Row);
	std::swap(m_Column, matrix.m_Column);
}
std::size_t Matrix::GetRow() const noexcept {
	return m_Row;
}
std::size_t Matrix::GetColumn() const noexcept {
	return m_Column;
}

Matrix Transpose12(Matrix& matrix) {
	float matrix1[1][2] = { matrix(0,0),matrix(0,1) };
	float matrix2[1][4] = { 0,0 };

	__m128 a = _mm_loadu_ps(&matrix1[0][0]);

	__m128 c = _mm_loadu_ps(&matrix2[0][0]);

	__m128 aa = _mm_unpacklo_ps(a, c);
	__m128 bb = _mm_unpackhi_ps(aa, c);

	Matrix out({ aa.m128_f32[0] , bb.m128_f32[0] }, 2, 1);

	return out;
}

Matrix Transpose24(Matrix& matrix) {
	float matrix1[2][4] = { matrix(0,0),matrix(0,1),matrix(0,2),matrix(0,3),matrix(1,0),matrix(1,1),matrix(1,2),matrix(1,3) };
	float matrixset[1][4] = { 0,0,0,0 };

	__m128 a = _mm_loadu_ps(&matrix1[0][0]);
	__m128 b = _mm_loadu_ps(&matrix1[1][0]);
	__m128 c = _mm_loadu_ps(&matrixset[0][0]);

	__m128 aa = _mm_unpacklo_ps(a, c);
	__m128 bb = _mm_unpackhi_ps(a, c);
	__m128 cc = _mm_unpacklo_ps(b, c);
	__m128 dd = _mm_unpackhi_ps(b, c);


	__m128 aaa = _mm_unpacklo_ps(aa, cc);
	__m128 bbb = _mm_unpackhi_ps(aa, cc);
	__m128 ccc = _mm_unpacklo_ps(bb, dd);
	__m128 ddd = _mm_unpackhi_ps(bb, dd);

	Matrix out({ aaa.m128_f32[0] , aaa.m128_f32[1] ,bbb.m128_f32[0] ,bbb.m128_f32[1],ccc.m128_f32[0],ccc.m128_f32[1],ddd.m128_f32[0] ,ddd.m128_f32[1] }, 4, 2);

	return out;
}

Matrix add_124(Matrix& matrixb, Matrix& matrixc) {
	float matrix1[2][4] = { matrixc(0,0),matrixc(0,1),matrixc(0,2),matrixc(0,3),matrixc(1,0),matrixc(1,1),matrixc(1,2),matrixc(1,3) };
	float matrix2[1][2] = { matrixb(0,0),matrixb(0,1) };

	__m128 b = _mm_load1_ps(&matrix2[0][0]);
	__m128 b1 = _mm_load1_ps(&matrix2[0][1]);
	__m128 c = _mm_loadu_ps(&matrix1[0][0]);
	__m128 d = _mm_loadu_ps(&matrix1[1][0]);

	__m128 mul1 = _mm_mul_ps(b, c);
	__m128 mul2 = _mm_mul_ps(b1, d);

	__m128 add1 = _mm_add_ps(mul1, mul2);

	Matrix out({ add1.m128_f32[0] , add1.m128_f32[1] ,add1.m128_f32[2] ,add1.m128_f32[3] }, 1, 4);

	return out;
}

Matrix add_142(Matrix& matrixb, Matrix& matrixc) {

	float matrix1[4][2] = { matrixc(0,0),matrixc(1,0),matrixc(2,0),matrixc(3,0),matrixc(0,1),matrixc(1,1),matrixc(2,1),matrixc(3,1) };
	float matrix2[1][4] = { matrixb(0,0),matrixb(0,1),matrixb(0,2),matrixb(0,3) };

	__m128 b = _mm_loadu_ps(&matrix1[0][0]);
	__m128 b1 = _mm_loadu_ps(&matrix1[2][0]);
	__m128 c = _mm_loadu_ps(&matrix2[0][0]);

	__m128 mul1 = _mm_mul_ps(c, b);
	__m128 mul2 = _mm_mul_ps(c, b1);

	float sum1 = mul1.m128_f32[0] + mul1.m128_f32[1] + mul1.m128_f32[2] + mul1.m128_f32[3];
	float sum2 = mul2.m128_f32[0] + mul2.m128_f32[1] + mul2.m128_f32[2] + mul2.m128_f32[3];

	Matrix out({ sum1,sum2 }, 1, 2);
	return out;
}

Matrix add_214(Matrix& matrixb, Matrix& matrixc) {

	float matrix1[1][4] = { matrixc(0,0),matrixc(0,1),matrixc(0,2),matrixc(0,3) };
	float matrix2[2][1] = { matrixb(0,0),matrixb(1,0) };

	__m128 b = _mm_load1_ps(&matrix2[0][0]);
	__m128 b1 = _mm_load1_ps(&matrix2[1][0]);
	__m128 c = _mm_loadu_ps(&matrix1[0][0]);

	__m128 mul1 = _mm_mul_ps(b, c);
	__m128 mul2 = _mm_mul_ps(b1, c);

	Matrix out({ mul1.m128_f32[0] , mul1.m128_f32[1] ,mul1.m128_f32[2] ,mul1.m128_f32[3],mul2.m128_f32[0] , mul2.m128_f32[1] ,mul2.m128_f32[2] ,mul2.m128_f32[3] }, 2, 4);

	return out;
}


Matrix add_141(Matrix& matrixb, Matrix& matrixc) {

	float matrix1[1][4] = { matrixb(0,0),matrixb(0,1),matrixb(0,2),matrixb(0,3) };
	float matrix2[1][1] = { matrixc(0,0) };

	__m128 b = _mm_loadu_ps(&matrix1[0][0]);
	__m128 c = _mm_load1_ps(&matrix2[0][0]);

	__m128 mul1 = _mm_mul_ps(b, c);
	float sum1 = mul1.m128_f32[0] + mul1.m128_f32[1] + mul1.m128_f32[2] + mul1.m128_f32[3];

	Matrix out({ sum1 }, 1, 1);

	return out;
}

void Matrix::HadamardProduct(const Matrix& matrix) noexcept {
	assert(m_Row == matrix.m_Row);
	assert(m_Column == matrix.m_Column);
	for (std::size_t i = 0; i < m_Data.size(); ++i) {
		m_Data[i] *= matrix.m_Data[i];
	}
}

Matrix operator*(double value, const Matrix& matrix) {
	return Matrix(matrix) *= value;

}
std::ostream& operator<<(std::ostream& stream, const Matrix& matrix) {

	stream << '[';

	bool isFirst = true;
	for (std::size_t i = 0; i < matrix.GetRow(); ++i) {
		if (isFirst) {
			isFirst = false;
		}
		else {
			stream << "\n ";
		}
		for (std::size_t j = 0; j < matrix.GetColumn(); ++j) {
			stream << ' ' << matrix(i, j);
		}
	}

	return stream << " ]";
}

//--------------------------acti------------------------------------------//



ActivationFunction::ActivationFunction(Matrix(*function)(const Matrix&), Matrix(*derivative)(const Matrix&)) noexcept
	: m_Function(function), m_Derivative(derivative) {
}

Matrix ActivationFunction::operator()(const Matrix& x) const {
	return m_Function(x);
}

Matrix ActivationFunction::GetGradient(const Matrix& x) const {
	return m_Derivative(x);
}

static inline __m128 FastExpSse(__m128 x)
{
	__m128 a = _mm_set1_ps(12102203.2f); // (1 << 23) / ln(2)
	__m128i b = _mm_set1_epi32(127 * (1 << 23) - 486411);
	__m128  m87 = _mm_set1_ps(-87);
	// fast exponential function, x should be in [-87, 87]
	__m128 mask = _mm_cmpge_ps(x, m87);

	__m128i tmp = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(a, x)), b);
	return _mm_and_ps(_mm_castsi128_ps(tmp), mask);
}


namespace {
	Matrix SigmoidF(const Matrix& x) {
		Matrix result(x.GetRow(), x.GetColumn());

		__m128 v[1];
		__m128 op1 = _mm_set1_ps(1);

		v[0] = _mm_set_ps(-x(0, 0), -x(0, 1), -x(0, 2), -x(0, 3));
		v[0] = _mm_add_ps(FastExpSse(v[0]), op1);
		v[0] = _mm_set_ps(1 / v[0].m128_f32[0], 1 / v[0].m128_f32[1], 1 / v[0].m128_f32[2], 1 / v[0].m128_f32[3]);

		for (int i = 0; i < 4; ++i) {
			result(0, i) = v[0].m128_f32[i];

		}

		return result;
	}

	Matrix SigmoidD(const Matrix& x) {
		Matrix result(x.GetRow(), x.GetColumn());
		Matrix func = SigmoidF(x);
		for (std::size_t i = 0; i < x.GetRow(); ++i) {
			for (std::size_t j = 0; j < x.GetColumn(); ++j) {
				result(i, j) = func(i, j) * (1 - func(i, j));
			}
		}
		return result;
	}
	Matrix ReLUF(const Matrix& x) {
		Matrix result(x.GetRow(), x.GetColumn());


		__m128 v[1];
		__m128 op1 = _mm_set1_ps(1);

		v[0] = _mm_set_ps(x(0, 0), x(0, 1), x(0, 2), x(0, 3));
		v[0] = _mm_max_ps(op1, v[0]);

		for (int i = 0; i < 4; ++i) {
			result(0, i) = v[0].m128_f32[i];

		}

		return result;
	}
	Matrix ReLUD(const Matrix& x) {
		Matrix result(x.GetRow(), x.GetColumn());
		for (std::size_t i = 0; i < x.GetRow(); ++i) {
			for (std::size_t j = 0; j < x.GetColumn(); ++j) {
				result(i, j) = x(i, j) >= 0 ? 1 : 0;
			}
		}
		return result;
	}
}

const ActivationFunction Sigmoid(SigmoidF, SigmoidD);
const ActivationFunction ReLU(ReLUF, ReLUD);


//--------------------------neuron----------------------------------------//



WeightedNeuron::WeightedNeuron(std::size_t inputSize)
	: m_W(1, inputSize) {
	std::random_device rd;
	std::mt19937_64 mt(rd());


	uniform_real_distribution<> dist(0.0, 1.0);

	for (std::size_t i = 0; i < inputSize; ++i) {
		while ((m_W(0, i) = dist(mt)) == 0);
	}
	while ((m_B = dist(mt)) == 0);
}

Matrix WeightedNeuron::Forward(Matrix& x) {

	m_X = x;
	m_U = add_124(m_W, x);


	__m128 v, v1;
	__m128 op1 = _mm_set_ps(m_B, m_B, m_B, m_B);
	v = _mm_set_ps(m_U(0, 0), m_U(0, 1), m_U(0, 2), m_U(0, 3));

	v1 = _mm_add_ps(v, op1);

	Matrix m_UU({ v1.m128_f32[3] ,v1.m128_f32[2] ,v1.m128_f32[1] ,v1.m128_f32[0] }, 1, 4);

	m_U = m_UU;
	return m_U;
}

Matrix WeightedNeuron::Backward(Matrix& d) {
	Matrix xT(m_X);
	Matrix newTr(4, 2);
	Matrix newTr2(2, 1);

	newTr = Transpose24(m_X);
	m_DW = add_142(d, newTr);

	Matrix a(1, m_X.GetColumn(), 1);
	m_DB = add_141(d, a);
	Matrix wT(m_W);
	newTr2 = Transpose12(m_W);

	return add_214(newTr2, d);
}
void WeightedNeuron::Update(double alpha) {
	m_W -= alpha * m_DW;
	m_B -= alpha * m_DB(0, 0);

}

ActivationNeuron::ActivationNeuron(ActivationFunction function)
	: m_F(function) {
}

Matrix ActivationNeuron::Forward(const Matrix& x) {
	m_U = x;
	m_Y = m_F(x);
	return m_Y;
}
Matrix ActivationNeuron::Backward(const Matrix& d) {
	Matrix up = m_F.GetGradient(m_U);
	up.HadamardProduct(d);
	return up;
}

//-----------------loss-------------------------------------------//



LossFunction::LossFunction(double(*function)(const Matrix&, const Matrix&), Matrix(*derivative)(const Matrix&, const Matrix&)) noexcept
	: m_Function(function), m_Derivative(derivative) {
}

double LossFunction::operator()(const Matrix& y, const Matrix& a) const {
	return m_Function(y, a);
}

Matrix LossFunction::GetGradient(const Matrix& y, const Matrix& a) const {
	return m_Derivative(y, a);
}

namespace {
	double MSEF(const Matrix& y, const Matrix& a) {
		double result = 0.0;
		Matrix ya;

		float av[2][4];
		__m128 v[2];
		v[0] = _mm_set_ps(y(0, 0), y(0, 1), y(0, 2), y(0, 3));
		v[1] = _mm_set_ps(a(0, 0), a(0, 1), a(0, 2), a(0, 3));


		v[0] = _mm_sub_ps(v[0], v[1]);

		v[0] = _mm_mul_ps(v[0], v[0]);

		result = v[0].m128_f32[0] + v[0].m128_f32[1] + v[0].m128_f32[2] + v[0].m128_f32[3];

		result /= 4;
		return result;
	}
	Matrix MSED(const Matrix& y, const Matrix& a) {

		Matrix ya = y - a;

		ya = ya *(2.0 / 4);

		return ya;
	}
}

const LossFunction MSE(MSEF, MSED);


int main() {

	clock_t start1, end1;
	float res1;
	int i;

	WeightedNeuron wn(2);
	ActivationNeuron an(Sigmoid);

	Matrix input({ 0, 1, 0, 1,
		0, 0, 1, 1 }, 2, 4);
	Matrix output({ 0, 1, 1, 1 }, 1, 4);

	start1 = clock();

	for (int i = 0; i < 1000; ++i) {
		cout << "1" << endl;
		Matrix y = wn.Forward(input);
		y = an.Forward(y);
		std::cout << "Epoch " << i + 1 << ": MSE " << MSE(y, output) << '\n';
		Matrix d = an.Backward(MSE.GetGradient(y, output));
		wn.Backward(d);
		wn.Update(0.3);
	}

	Matrix y = wn.Forward(input);
	y = an.Forward(y);
	cout << y << '\n';

	end1 = clock();
	res1 = (float)(end1 - start1) / CLOCKS_PER_SEC;
	printf(" SSE함수 소요된 시간 : %.3f \n", res1);
	return 0;
}
