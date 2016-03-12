#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;


float logsig(float x) {
    return 1.0 / (1.0 + pow(2.7182818284590452353602874713527, -x ) );
}

class Matrix {

    size_t rows, cols;
    std::vector< std::vector<float> > val;

public:

    Matrix() {}

    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        (*this).allocate();
    }


    void allocate(size_t r, size_t c) {
        rows = r; cols = c;
        val.resize(rows);
        for (size_t y = 0; y < rows; y++) {
            val[y].resize(cols);

            for (size_t x = 0; x < cols; x++) {
                val[y][x] = (float)rand()/(float)(RAND_MAX) - 0.5;
            }
        }
    };

    void allocate() {
        val.resize(rows);
        for (size_t y = 0; y < rows; y++) {
            val[y].resize(cols);

            for (size_t x = 0; x < cols; x++) {
                val[y][x] = (float)rand()/(float)(RAND_MAX) - 0.5;
            }
        }
    };

    std::vector<float> & operator [] (size_t idx) {
        return val[idx];
    }

    const std::vector<float> & operator [](size_t idx) const {
        return val[idx];
    }


    Matrix & operator += (const Matrix &rhs) {
        for (size_t y = 0; y < rows; y++) {
            for (size_t x = 0; x < cols; x++) {
                (*this)[y][x] += rhs[y][x];
            }
        }

        return *this;
    }

    const Matrix operator+(const Matrix &other) const {
        Matrix result = *this;
        result += other;
        return result;
    }


    Matrix & operator *= (const Matrix &rhs) {
        std::vector<float> row_vals;
        for (size_t y = 0; y < rows; y++) {
            for (size_t x = 0; x < cols; x++) {
                (*this)[y][x] = (*this)[y][x] * rhs[y][x];
            }
        }
        return *this;
    }


    const Matrix operator*(const Matrix &other) const {

        Matrix result = *this;
        result *= other;
        return result;
    }

    const Matrix multiply(const Matrix &other) const {

        Matrix result (rows, other.cols);

        for (size_t y = 0; y < rows; y++) {
            for (size_t x = 0; x < other.cols; x++) {
                result[y][x] = 0;
                for (size_t i = 0; i < cols; i++) {
                    result[y][x] += (*this)[y][i] * other[i][x];
                }
            }
        }
        return result;
    }

    Matrix pwise_fxn(float (*fxn)(float)) {
        Matrix result = (*this);
        for (size_t y = 0; y < rows; y++) {
            for (size_t x = 0; x < cols; x++) {
                result[y][x] = fxn((*this)[y][x]);
            }
        }
        return result;
    }

    void cat(Matrix B, int dim) {
        if ( dim == 1 ) {
            val.insert(val.end(), B.val.begin(), B.val.end());
            rows = rows + B.rows;
        } else if (dim == 2) {
            for (size_t y = 0; y < rows; y++) {
                val[y].insert(val[y].end(), B[y].begin(), B[y].end());
            }
            cols = cols + B.cols;
        }
    }


    void report_size () {
        cout << (*this).rows << " " << (*this).cols << "\n";
    }

};

class NN_layer {

    Matrix weight;
    Matrix bias;
    Matrix input;
    Matrix output;

public:

    NN_layer() {}

    NN_layer(size_t input_length, size_t output_length) {
        weight.allocate(input_length, output_length);
        bias.allocate(1, output_length);
    }

    run(Matrix input)

};

class Memory_cell {

    Matrix forget_W, input_W, cand_W, out_W;

    Matrix forget_bias, input_bias, cand_bias, out_bias;

    Matrix cell_in, cell_out, h_in, h_out, x_in, forget_out, gate_in, input_act, cand_act, output_act, cell_tanh;

    //Matrix forget_preact;

public:

    Memory_cell(size_t input_length, size_t output_length) {

        size_t gate_in_length = input_length + output_length;

        forget_W.allocate(gate_in_length, output_length);
        input_W.allocate(gate_in_length, output_length);
        cand_W.allocate(gate_in_length, output_length);
        out_W.allocate(gate_in_length, output_length);

        cell_in.allocate(1, output_length);
        h_in.allocate(1, output_length);
        h_out.allocate(1, output_length);


        forget_bias.allocate(1, output_length);
        input_bias.allocate(1, output_length);
        cand_bias.allocate(1, output_length);
        out_bias.allocate(1, output_length);

        gate_in.allocate(1, gate_in_length);
    }


    Matrix forward_pass( Matrix x_in ) {

        gate_in = x_in;
        gate_in.cat(h_in, 2);


        forget_out = gate_in.multiply(forget_W) + forget_bias;

        input_act = gate_in.multiply(input_W) + input_bias;

        cand_act = gate_in.multiply(cand_W) + cand_bias;

        output_act = gate_in.multiply(out_W) + out_bias;


        cell_out = cand_act.pwise_fxn(&tanhf) * input_act.pwise_fxn(&logsig) + cell_in * forget_out.pwise_fxn(&logsig);

        cell_tanh = cell_out;
        cell_tanh.pwise_fxn(&tanhf);

        h_out = output_act.pwise_fxn(&logsig) * cell_tanh;

        return h_out;
    }

    void backward_pass( Matrix err ) {

        dout_d



        gate_in = x_in;
        gate_in.cat(h_in, 1);


        forget_out = forget_W.multiply(gate_in) + forget_bias;

        input_act = input_W.multiply(gate_in) + input_bias;

        cand_act = cand_W.multiply(gate_in) + cand_bias;

        output_act = out_W.multiply(gate_in) + out_bias;


        cell_out = cand_act.pwise_fxn(&tanhf) * input_act.pwise_fxn(&logsig) + cell_in * forget_out.pwise_fxn(&logsig);

        cell_tanh = cell_out;
        cell_tanh.pwise_fxn(&tanhf);

        h_out = output_act.pwise_fxn(&logsig) * cell_tanh;

    }
};


int main()
{


    srand (time(NULL));

    Matrix x_in (10, 1);
    Memory_cell mc (10, 5);
    mc.forward_pass(x_in);

    /*
    Matrix M (5,5);
    Matrix N (5,5);


    Matrix Z = M + N;
    cout << M[2][2] << "\n";
    cout << N[2][2] << "\n";
    cout << Z[2][2] << "\n";
    */
    return 0;
}
