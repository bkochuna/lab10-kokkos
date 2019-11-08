/*************************************************
* Laplace Serial C++ Version
*
* Temperature is initially 0.0
* Boundaries are as follows:
*
*      0         T         0
*   0  +-------------------+  0
*      |                   |
*      |                   |
*      |                   |
*   T  |                   |  T
*      |                   |
*      |                   |
*      |                   |
*   0  +-------------------+ 100
*      0         T        100
*
* Derived from program provided by:
*  John Urbanic, PSC 2014
*
************************************************/

#include <sys/time.h>
#include <Kokkos_Core.hpp>

// Error allowed in temperature
#define MAX_TEMP_ERROR 0.01
#define MAX_ITERATIONS 4000

// Type alias for plate temperatures
typedef Kokkos::View<double**> temperature;

// Helper routines
void initialize(const unsigned ROWS, const unsigned COLUMNS, temperature T);
void track_progress(const unsigned ROWS, const unsigned COLUMNS,
                    const unsigned iter, const temperature T);

using namespace std;

int main(int argc, char* argv[]){
  // Read input
  unsigned ROWS = 0;
  unsigned COLUMNS = 0;
  if(argc >= 3 ){
    ROWS = static_cast<unsigned>(atoi(argv[1]));
    COLUMNS = static_cast<unsigned>(atoi(argv[2]));
  } else {
    printf("Wrong number of inputs!\n");
    return -1;
  }

  Kokkos::initialize(argc, argv);
  {
    // Timer products
    struct timeval begin, end;
    gettimeofday(&begin, NULL); // starting time

    // Allocate our containers
    temperature T("T", ROWS+2, COLUMNS+2);
    temperature T_prev("prev", ROWS+2, COLUMNS+2);

    // Initialize the conditions
    initialize(ROWS, COLUMNS, T_prev);

    // Iterations
    unsigned iter;
    double dT = 100.0;
    double diff = 0.0;

    while(dT > MAX_TEMP_ERROR && iter <= MAX_ITERATIONS){
      // Main calculation - average 4 neighbors
      // This should be parallelized
      for(unsigned row = 1; row <= ROWS; ++row){
        for(unsigned col = 1; col <= COLUMNS; ++col){
          T(row, col) = 0.25 * (
              T_prev(row+1, col) + T_prev(row-1, col)
            + T_prev(row, col+1) + T_prev(row, col-1)
          );
        }
      }

      // Copy the grid to the old grid and determine largest temperature change
      dT = 0.0;
      for(unsigned row = 1; row <= ROWS; ++row){
        for(unsigned col = 1; col <= COLUMNS; ++col){
          diff = std::abs(T(row, col) - T_prev(row, col));
          dT = std::max(dT, diff);
          T_prev(row, col) = T(row, col);
        }
      }

      // Track progress periodically
      if((iter % 100) == 0) track_progress(ROWS, COLUMNS, iter, T);
      // Increment counter
      ++iter;
    }

    // Ending time
    gettimeofday(&end, NULL);

    // Compute calculation time
    double time = 1.0 * (end.tv_sec - begin.tv_sec) +
                  1.0e-6 * (end.tv_usec - begin.tv_usec);

    printf("\n");
    printf("Maximum error at iteration %d was %lf\n", iter-1, dT);
    printf("Total runtime was %lf seconds.\n", time);
  }
  Kokkos::finalize();

  return 0;
}

// Initialize the temperature of the grid
// All zero except boundary conditions
void initialize(const unsigned ROWS, const unsigned COLUMNS, temperature prev){
  // Initialize all to zero save last row/column
  for(unsigned row = 0; row < ROWS+1; ++row){
    for(unsigned col = 0; col < COLUMNS+1; ++col){
      prev(row, col) = 0.0;
    }
  }
  // Initialize boundary conditions
  // Left side is set to zero
  // Right side linearly increases from 0 to 100
  for(unsigned row = 0; row <= ROWS+1; ++row){
    prev(row, 0) = 0.0;
    prev(row, COLUMNS+1) = (100.0/ROWS)*row;
  }
  // Top side is set to zero
  // Bottom side linearly increases from 0 to 100
  for(unsigned col = 0; col < COLUMNS+1; ++col){
    prev(0, col) = 0.0;
    prev(ROWS+1, col) = (100.0/COLUMNS)*col;
  }
}

// Track progress
void track_progress(const unsigned ROWS, const unsigned COLUMNS,
                    const unsigned iter, temperature T){
  printf("---------- Iteration number: %d ------------\n", iter);
  for(unsigned i = ROWS-5; i <= ROWS; i++){
    printf("[%d,%d]: %5.2f  ", i, i, T(i,i) );
  }
  printf("\n");
}
