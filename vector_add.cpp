#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace std;

void initArray(int arr[], int n)
{
    for (int i = 0; i < n; ++i)
    {
        arr[i] = rand() % n;
    }
}

void printArray(int arr[], int n)
{
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << " ";
    }
    cout << "\n";
}

void verfyOutput(int *a, int *b, int *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        assert(a[i] + b[i] == c[i]);
    }
}

int main()
{
    const int N = 1e5;
    int a[N], b[N], c[N];

    // Initialize input arrays
    initArray(a, N);
    initArray(b, N);

    // Create a SYCL queue
    queue q(default_selector{});

    // Create buffers for input and output arrays
    buffer<int, 1> a_buf(a, range<1>(N));
    buffer<int, 1> b_buf(b, range<1>(N));
    buffer<int, 1> c_buf(c, range<1>(N));

    // Submit a kernel to compute c = a + b
    q.submit([&](handler &h)
             {
                auto a_acc = a_buf.get_access<access::mode::read>(h);
                auto b_acc = b_buf.get_access<access::mode::read>(h);
                auto c_acc = c_buf.get_access<access::mode::write>(h);

                h.parallel_for(range<1>{N}, [=](id<1> i) {
                  c_acc[i] = a_acc[i] + b_acc[i];
                }); });

    // Wait for the kernel to complete execution
    q.wait();

    // Verify the output
    verfyOutput(a, b, c, N);

    // cout << "Array a:- ";
    // printArray(a, N);
    // cout << "Array b:- ";
    // printArray(b, N);
    // cout << "Array c:- ";
    // printArray(c, N);
    cout << "Success!\n";
    return 0;
}
