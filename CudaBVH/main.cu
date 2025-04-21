#include "lbvh.cuh"
#include "CudaHeader.cuh"

#include <random>
#include <vector>
#include <thrust/random.h>


#include "GPU4UETest.cuh"

int main()
{
	std::cout << "START." << std::endl;

	//GPU4UE::Test1();
	GPU4UE::Test2();
	//GPU4UE::Test3();

	std::cout << "END." << std::endl;


	return 0;
}