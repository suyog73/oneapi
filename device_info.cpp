#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace std;

int main()
{
    queue q;
    cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";
    return 0;
}
