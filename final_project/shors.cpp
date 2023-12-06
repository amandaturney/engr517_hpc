#define _USE_MATH_DEFINES

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
int numMeasuredQbits = 3; //qubits 0-2 are measured at end
int numCountingQbits = numMeasuredQbits -1; // 2 qubits are used for 'counting' while 1 represents our state psi
std::complex<double> registerStateAmplitudes [8] = {0.0};  // holds amplitudes of all possible states across numMeasuredQubits


// Define custom structs/typedefs
struct qubitsRegister{ 
    char state[7];
    std::complex<double> amplitude;
};

typedef void (*quantumCircuit)( qubitsRegister *qubitsRegister );


// Some general helper methods
int binaryToDecimal(const char* binaryString) {
    int decimal = 0;
    int n = strlen(binaryString) - 1;

    // Right-most bit is smallest
    for (int i = n; i >= 0 ; --i) {
        if (binaryString[i] == '1') {
            decimal+= pow(2.0, (n-i));
        }
    }

    return decimal;
}

int gcd(int a, int b) {
    // Greatest common divisor
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}



/////// Quantum Circuit & Operator Functions ////////
void applyCircuit(qubitsRegister *qubitsRegister, int circuitOffset, int circuitLength, const quantumCircuit* circuit) {
    // First apply the register to the circuit
    for(int i=circuitOffset; i<circuitLength; ++i) {
        circuit[i](qubitsRegister);
    }

    // Multiple paths could have resulted in same state, so reduce amplitudes for each state for the measured qubits only
    char measuredState[numMeasuredQbits+1];
    memcpy(measuredState, qubitsRegister->state, numMeasuredQbits); // no offset as we always take measured qubits from top
    measuredState[numMeasuredQbits] = 0; // terminate

    int stateDecimal = binaryToDecimal(measuredState);
    registerStateAmplitudes[stateDecimal] += qubitsRegister->amplitude;

    return;
}

void CNOT(int controlQb, int targetQb, qubitsRegister *qubitsRegister) {
    // If control qubit is 1, then flip target qubit
    if (qubitsRegister->state[controlQb] == '1') {
        qubitsRegister->state[targetQb] = qubitsRegister->state[targetQb] == '0' ? '1' : '0';
    }
}

void NOT(int targetQb, qubitsRegister *qubitsRegister) {
    // Flip the target qubit's state
    qubitsRegister->state[targetQb] = qubitsRegister->state[targetQb] == '0' ? '1' : '0';
}

void HADAMARD(int targetQb, qubitsRegister *qubitsRegister, int circuitOffset, int circuitLength, const quantumCircuit* circuit) {
    qubitsRegister->amplitude /= sqrt(2.0);

    struct qubitsRegister registerNew;
    strcpy(registerNew.state, qubitsRegister->state);
    registerNew.amplitude = qubitsRegister->amplitude;

    if (qubitsRegister->state[targetQb] == '0') {
        registerNew.state[targetQb] = '1';
    }
    else {
        registerNew.state[targetQb] = '0';
        qubitsRegister->amplitude *= -1.0;
    }

#pragma omp task
    applyCircuit(&registerNew, circuitOffset+1, circuitLength, circuit);
}

void CP(int controlQb, int targetQb, double phase, qubitsRegister *qubitsRegister) {
    // If control qubit is 1, then flip target qubit
    if (qubitsRegister->state[controlQb] == '1') {
        std::complex<double> i(0.0, 1.0);
        qubitsRegister->amplitude *= exp(i*phase);
    }
}

int measure(int randomNumber, int probabilityFactor) {
    // Store the states with non-zero amplitudes and their accumulated probabilities
    std::vector<int> accumulatedProbs = {0};
    std::vector<int> nonZeroProbStates;

    for (int i = 0; i<pow(2, numMeasuredQbits); ++i){
        // probability is amplitude squared, always a real number
        double probability = pow(registerStateAmplitudes[i], 2).real();

        if (probability > 0.0){
            accumulatedProbs.push_back(int(probabilityFactor*probability) + accumulatedProbs[i]);
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


/////// Shors Algorithm - QC Circuit ////////
quantumCircuit shors4xMod15[10] = {
    [](qubitsRegister *qubitsRegister) {HADAMARD(0, qubitsRegister, 0, 10, shors4xMod15);},
    [](qubitsRegister *qubitsRegister) {HADAMARD(1, qubitsRegister, 1, 10, shors4xMod15);},
    [](qubitsRegister *qubitsRegister) {HADAMARD(2, qubitsRegister, 2, 10, shors4xMod15);},
    [](qubitsRegister *qubitsRegister) {NOT(2, qubitsRegister);},
    [](qubitsRegister *qubitsRegister) {CNOT(2, 5, qubitsRegister);},
    [](qubitsRegister *qubitsRegister) {NOT(2, qubitsRegister);},
    [](qubitsRegister *qubitsRegister) {CNOT(2, 3, qubitsRegister);},
    [](qubitsRegister *qubitsRegister) {HADAMARD(0, qubitsRegister, 7, 10, shors4xMod15);},
    [](qubitsRegister *qubitsRegister) {CP(0, 1, -M_PI/2, qubitsRegister);},
    [](qubitsRegister *qubitsRegister) {HADAMARD(1, qubitsRegister, 9, 10, shors4xMod15);}
    
};

int getPeriod(int probabilityFactor) {
    // First, set the registerStateAmplitudes to zero which is needed when rerunning
    for (int i = 0; i<8; ++i) {
        registerStateAmplitudes[i] = 0.0;
    }

    // Then generate a random number which will be used to randomly measure the final state
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_int_distribution<int> distribution(0, probabilityFactor);
    int randomNumber = distribution(generator);


    // Set up our initial qubits register with all qubits in state 0
    struct qubitsRegister registerInit;
    strcpy(registerInit.state, "000000");
    registerInit.amplitude = 1.0;


    // Run the quantum circuit
#pragma omp parallel
#pragma omp single
#pragma omp taskgroup task_reduction(+:registerStateAmplitudes)
    applyCircuit(&registerInit, 0, 10, shors4xMod15);


    // Measure the circuits state and convert into a period
    int measuredState = measure(randomNumber, probabilityFactor);
    int period = measuredState != 0 ? pow(2, numCountingQbits) / (measuredState * 2) : 0;

    return period;

}



int main(int argc, char** argv)
{
    // Starting Shors Algorithm
    int N = 15;
    fmt::print("Performing Shor's Algorithm to factor {} with {} threads\n", N, omp_get_max_threads());



    // First, use quantum algorithm to find period of function. This algorithm
    // guaruntees correct answer at least 1/2 the time so run until we get valid/non-zero answer
    int probabilityFactor = 1000;
    int period = 0;
    int numTries = 0;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    while (period == 0){
        period = getPeriod(probabilityFactor);
        numTries++;
    }


    // Next part of the algorithm is classical; plug in the period for the factors of N
    int primeFactor1 = gcd(pow(4, int(period/2))-1, N);
    int primeFactor2 = gcd(pow(4, int(period/2))+1, N);


    // Finish timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;
    double ns = time.count();                   // time in nanoseconds


    // Print back out the results
    fmt::print("The estimate of the factors of {} are {}, {}.\n", N, primeFactor1, primeFactor2);
    fmt::print("Number of tries: {}, using {} threads, time spent processing: {} ns\n", numTries, omp_get_max_threads(), ns);
    return 0;
}