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
	Matrix operator*(double value) const;
	Matrix operator*(const Matrix& matrix) const;
	Matrix& operator+=(const Matrix& matrix) noexcept;
	Matrix& operator-=(const Matrix& matrix) noexcept;
	Matrix& operator*=(double value) noexcept;
	Matrix& operator*=(const Matrix& matrix);

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
	Matrix Forward(const Matrix& x);
	Matrix Backward(const Matrix& d);
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
Matrix Matrix::operator*(double value) const {

	return Matrix(*this) *= value;
}
Matrix Matrix::operator*(const Matrix& matrix) const {
	
	assert(m_Column == matrix.m_Row);

	Matrix result(m_Row, matrix.m_Column);
	

	for (std::size_t i = 0; i < m_Row; ++i) {
		for (std::size_t j = 0; j < matrix.m_Column; ++j) {
			for (std::size_t k = 0; k < m_Column; ++k) {
			
				result(i, j) += (*this)(i, k) * matrix(k, j);
			}
		}
	}

	return result;
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
		//cout << "1" << endl;
	}

	return *this;
}
Matrix& Matrix::operator*=(const Matrix& matrix) {
//	cout << "2" << endl;
	return *this = (*this) * matrix;
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

void Matrix::Transpose() {
	Matrix temp(m_Column, m_Row);
	for (std::size_t i = 0; i < m_Row; ++i) {
		for (std::size_t j = 0; j < m_Column; ++j) {
			temp(j, i) = (*this)(i, j);
		}
	}
	Swap(temp);
}
void Matrix::HadamardProduct(const Matrix& matrix) noexcept {
	assert(m_Row == matrix.m_Row);
	assert(m_Column == matrix.m_Column);
	//cout << "5" << endl;
	for (std::size_t i = 0; i < m_Data.size(); ++i) {
		m_Data[i] *= matrix.m_Data[i];
	}
}

Matrix operator*(double value, const Matrix& matrix) {
	return Matrix(matrix) *= value;
//	cout << "3" << endl;
}
std::ostream& operator<<(std::ostream& stream, const Matrix& matrix) {
//	cout << "4 //	" << endl;
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

namespace {
	Matrix SigmoidF(const Matrix& x) {
		Matrix result(x.GetRow(), x.GetColumn());
		for (std::size_t i = 0; i < x.GetRow(); ++i) {
			for (std::size_t j = 0; j < x.GetColumn(); ++j) {
				result(i, j) = 1 / (1 + std::exp(-x(i, j)));
			}
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
		for (std::size_t i = 0; i < x.GetRow(); ++i) {
			for (std::size_t j = 0; j < x.GetColumn(); ++j) {
				result(i, j) = std::max(0.0, x(i, j));
			}
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

Matrix WeightedNeuron::Forward(const Matrix& x) {

	m_X = x;
	m_U = m_W * x;

	for (std::size_t i = 0; i < m_U.GetColumn(); ++i) {
		m_U(0, i) += m_B;
	}
	return m_U;
}
Matrix WeightedNeuron::Backward(const Matrix& d) {
	Matrix xT(m_X);
	xT.Transpose();
	m_DW = d * xT;
	m_DB = d * Matrix(1, m_X.GetColumn(), 1);

	Matrix wT(m_W);
	wT.Transpose();
	return wT * d;
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
		Matrix ya = y - a;
		for (std::size_t i = 0; i < ya.GetColumn(); ++i) {
			result += std::pow(ya(0, i), 2);
		}
		result /= ya.GetColumn();
		return result;
	}
	Matrix MSED(const Matrix& y, const Matrix& a) {
		Matrix ya = y - a;
		ya *= (2.0 / y.GetColumn());
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
	Matrix output({ 0, 0, 1, 1 }, 1, 4);

	start1 = clock();

	for (int i = 0; i < 1000; ++i) {
		cout << "1"<< endl;
		Matrix y = wn.Forward(input);
		//cout << y.GetRow() << "/////////" << y.GetColumn() << endl;
		y = an.Forward(y);
		//cout << y.GetRow() << "/////////" << y.GetColumn()<<endl;
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
	printf(" 일반함수 소요된 시간 : %.3f \n", res1);
	return 0;
}