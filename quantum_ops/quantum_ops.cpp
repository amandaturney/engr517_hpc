#define _USE_MATH_DEFINES

#include <docopt.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <complex>
#include <random>
#include <set>
#include <string>
#include <vector>


// Custom reduction for complex data type
#pragma omp declare	reduction(+ : std::complex<double> : omp_out += omp_in ) initializer( omp_priv = omp_orig )

// Global variables
int n = 3;
int numStates = pow(2, 4);
std::complex<double> registerStateAmplitudes [16] = {0.0};


// Define custom structs/typedefs
struct qubitsRegister{ 
    char state[9];
    std::complex<double> amplitude;
};


typedef std::function<void(qubitsRegister *qRegister)> quantumOp;
typedef std::vector<quantumOp> quantumCircuit;


// Some general helper methods
int binaryToDecimal(const char* binaryString) {
    int decimal = 0;
    int n = strlen(binaryString) - 1;

    // Right-most bit is smallest
    for (int i = n; i >= 0; --i) {
        if (binaryString[i] == '1') {
            decimal+= pow(2.0, (n-i));
        }
    }

    return decimal;
}

void decimalToBinary(int decimal, int n, bool pad, char* binaryString) {

    for (int i = 0; i < n; ++i) {
        int sig = pow(2.0, (n-i-1));
        if (decimal - sig >= 0.0){
            binaryString[i] = '1';
            decimal -= sig;
        }
        else {
            binaryString[i] = '0';
        }
    }

    if (pad) {
        binaryString[n] = '0';
        binaryString[n+1] = 0;
    }
    else{
        binaryString[n] = 0;
    }
}


/////// Quantum Circuit & Operator Functions ////////
void applyCircuit(qubitsRegister *qRegister, int circuitOffset, quantumCircuit* circuit) {
    for(int i=circuitOffset; i<circuit->size(); ++i) {
        circuit->at(i)(qRegister);
    }

    // Multiple paths could have resulted in same state, so reduce amplitudes for each state for the measured qubits only
    char measuredState[5];
    memcpy(measuredState, qRegister->state, 4); // no offset as we always take measured qubits from top
    measuredState[4] = 0; // terminate

    int stateDecimal = binaryToDecimal(measuredState);
    registerStateAmplitudes[stateDecimal] += qRegister->amplitude;

    return;
}

void CNOT(int controlQb, int targetQb, qubitsRegister *qRegister) {
    // If control qubit is 1, then flip target qubit
    if (qRegister->state[controlQb] == '1') {
        qRegister->state[targetQb] = qRegister->state[targetQb] == '0' ? '1' : '0';
    }
}

void NOT(int targetQb, qubitsRegister *qRegister) {
    // Flip the target qubit's state
    qRegister->state[targetQb] = qRegister->state[targetQb] == '0' ? '1' : '0';
}

void HADAMARD(int targetQb, qubitsRegister *qRegister, int circuitOffset, quantumCircuit* circuit) {
    qRegister->amplitude /= sqrt(2.0);

    struct qubitsRegister registerNew;
    strcpy(registerNew.state, qRegister->state);
    registerNew.amplitude = qRegister->amplitude;

    if (qRegister->state[targetQb] == '0') {
        registerNew.state[targetQb] = '1';
    }
    else {
        registerNew.state[targetQb] = '0';
        qRegister->amplitude *= -1.0;
    }

#pragma omp task
    applyCircuit(&registerNew, circuitOffset+1, circuit);
}

void CP(int controlQb, int targetQb, double phase, qubitsRegister *qRegister) {
    // If control and and target qubit is 1, then rotate
    if ((qRegister->state[controlQb] == '1') & (qRegister->state[targetQb] == '1')) {
        std::complex<double> i(0.0, 1.0);
        qRegister->amplitude *= exp(i*phase);
    }
}

void PHASE(int targetQb, double phase, qubitsRegister *qRegister) {
    if (qRegister->state[targetQb] == '1') {
        std::complex<double> i(0.0, 1.0);
        qRegister->amplitude *= exp(i*phase);
    }
}
 
void SWAP(int qb1, int qb2, qubitsRegister *qRegister) {
    std::swap(qRegister->state[qb1], qRegister->state[qb2]);
}

void _QFTRotations(int nQb, quantumCircuit* circuit) {
    if (nQb == 0) {
        return;
    }
    else {
        nQb--;
        int circuitSize = circuit->size();
        circuit->push_back(
            [nQb, circuitSize, circuit](qubitsRegister *qRegister) {HADAMARD(nQb, qRegister, circuitSize, circuit);}
        );
        fmt::print("Doing a hadamard on {}.\n", nQb);

        for (int i = 0; i < nQb; ++i) {
            circuit->push_back(
                [i, nQb](qubitsRegister *qRegister){CP(i, nQb, M_PI/(pow(2, (nQb-i))), qRegister);}
            );
            fmt::print("Doing a CP on {} and {} with phase pi/{}.\n", i, nQb, pow(2, (nQb-i)));
        }

        _QFTRotations(nQb, circuit);
    }
}

void QFT(int nQb, quantumCircuit* circuit) {
    // Performs Quantum Fourier Transform to transform state into fourier phases
    _QFTRotations(nQb, circuit);

    for (int i = 0; i < int(nQb/2); ++i) {
        circuit->push_back(
            [i, nQb](qubitsRegister *qRegister) {SWAP(i, (nQb-i-1), qRegister);}
        );
        fmt::print("Doing a swap on {} and {}.\n", i, (nQb-i-1));
    }
}

void _QFTInverse(int nQb, quantumCircuit* circuit) {
    for (int i = 0; i < nQb; ++i) {
        for (int j = 0; j < i; ++j) {
            double phase = -1*M_PI/(pow(2,(i-j)));
            circuit->push_back(
                [i, j, nQb, phase](qubitsRegister *qRegister){CP(i, j, phase, qRegister);}
            );
            fmt::print("Doing a CP on {} and {} with phase -pi/{}.\n", j, i, pow(2,(i-j)));
        }
        int circuitSize = circuit->size();
        circuit->push_back(
            [i, circuitSize, circuit](qubitsRegister *qRegister) {HADAMARD(i, qRegister, circuitSize, circuit);}
        );
        fmt::print("Doing a hadamard on {}.\n", i);
    }
}

void QFTDagger(int nQb, quantumCircuit* circuit) {
    // Performs Quantum Fourier Transform Inverse/Dagger to transform state from fourier phases
    // back to computational 
    for (int i = 0; i < int(nQb/2); ++i) {
        circuit->push_back(
            [i, nQb](qubitsRegister *qRegister) {SWAP(i, (nQb-i-1), qRegister);}
        );
        fmt::print("Doing a swap on {} and {}.\n", i, (nQb-i-1));
    }

    _QFTInverse(nQb, circuit);
}

void ADD(int nQb, quantumCircuit* circuit) {
    // Always assume that a starts at qubit 0 and b starts at qubit n/2
    for (int i = nQb; i > 0; --i){
        for (int j = i; j > 0; --j) {
            if (nQb-2 >= j - 1) {
                circuit->push_back(
                    [nQb, i, j](qubitsRegister *qRegister) {CP(nQb-i, nQb-1+j, 2*M_PI/(pow(2, (i-j+1))), qRegister);}
                );
                fmt::print("Doing a CP on {} and {} with phase {}.\n", nQb-i, nQb-1+j, 2*M_PI/(pow(2, (i-j+1))));
            }
        }
    }
}

void initiateRegister(int nQb, int offset, char* binaryString, quantumCircuit* circuit) {
    for (int i=0; i < nQb; ++i) {
        if (binaryString[i] == '1') {
            circuit->push_back(
                [i, offset](qubitsRegister *qRegister) {NOT(i+offset, qRegister);}
            );
            fmt::print("Doing a NOT on {}", i+offset);
        }
    }
}

int measure(int randomNumber, int probabilityFactor) {
    // Store the states with non-zero amplitudes and their accumulated probabilities
    std::vector<int> accumulatedProbs = {0};
    std::vector<int> nonZeroProbStates;

    for (int i = 0; i<numStates; ++i){
        // probability is amplitude squared, always a real number
        double probability = pow(registerStateAmplitudes[i].real(), 2) + pow(registerStateAmplitudes[i].imag(), 2);
        fmt::print("The probability of state {} is {}.\n", i, probability);

        if (probability > 0.0){
            accumulatedProbs.push_back(int(probabilityFactor*probability) + accumulatedProbs[accumulatedProbs.size()-1]);
            nonZeroProbStates.push_back(i);
        }
    }

    // Loop through non-zero probability states and check if random number falls within their
    // accumulated probability range; if so, return that state
    for (int i = 0; i<nonZeroProbStates.size(); ++i) {
        if (randomNumber >= accumulatedProbs[i] && randomNumber <= accumulatedProbs[i+1]) {
            return nonZeroProbStates[i];
        }
    }
    
    return -1;
}


constexpr const char USAGE[] =
 R"(quantum_ops: perform basic operations by quantum algorithms

  Usage:
      quantum_ops (-h | --help)
      quantum_ops --version
      quantum_ops <a> <b>

  Options:
     -h --help         This screen.
     --version         Print the version of this code.
)";

int main(int argc, char** argv)
{
    // First, set the registerStateAmplitudes to zero which is needed when rerunning
    for (int i = 0; i<numStates; ++i) {
        registerStateAmplitudes[i] = 0.0;
    }

    auto args = docopt::docopt(USAGE, {argv + 1, argv + argc});
    int a = args["<a>"].asLong();
    int b = args["<b>"].asLong();

    fmt::print("Performing the quantum adder on {} and {}.\n", a, b);

    // Convert to binary
    char aBinary[5];
    decimalToBinary(a, n+1, false, aBinary);
    char bBinary[4];
    decimalToBinary(b, n, false, bBinary);
    

    // Then generate a random number which will be used to randomly measure the final state
    int probabilityFactor = 1000;
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_int_distribution<int> distribution(0, probabilityFactor);
    int randomNumber = distribution(generator);

    struct qubitsRegister registerInit;
    strcpy(registerInit.state, "00000000");
    registerInit.amplitude = 1.0;

    quantumCircuit adder;
    initiateRegister(4, 0, aBinary, &adder);
    initiateRegister(4, 4, bBinary, &adder);
    QFT(4, &adder);
    for (int i = 0; i < 20000; ++i) {
        ADD(4, &adder);
    }
    QFTDagger(4, &adder);
    fmt::print("The size of the circuit is now: {}\n", adder.size());

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();


#pragma omp parallel
#pragma omp single
#pragma omp taskgroup task_reduction(+:registerStateAmplitudes)
    applyCircuit(&registerInit, 0, &adder);


    // Measure the circuits state and convert into a period
    int measuredState = measure(randomNumber, probabilityFactor);

    // Finish timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;
    double ns = time.count();                   // time in nanoseconds

    fmt::print("Omg did this work? The answer is: {}\n", measuredState);
    fmt::print("grep{},{}\n", omp_get_max_threads(), ns);
    return 0;
}