#include <AMReX.H>

#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#ifdef PELERAD_ENABLE_MPI
#include <mpi.h>
#endif

struct ScopeGuard
{
    ScopeGuard(int argc, char* argv[])
    {

#ifdef PELERAD_ENABLE_MPI
        MPI_Init(&argc, &argv);
#endif
        amrex::Initialize(argc, argv);
    }

    ~ScopeGuard()
    {
        amrex::Finalize();
#ifdef PELERAD_ENABLE_MPI
        MPI_Finalize();
#endif
    }
};

bool init_function() { return true; }

int main(int argc, char* argv[])
{
    ScopeGuard scope_guard(argc, argv);
    return boost::unit_test::unit_test_main(&init_function, argc, argv);
}

