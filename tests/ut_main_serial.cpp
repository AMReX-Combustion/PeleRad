#include <AMReX.H>

#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

struct ScopeGuard
{
    ScopeGuard(int argc, char* argv[]) { amrex::Initialize(argc, argv); }

    ~ScopeGuard() { amrex::Finalize(); }
};

bool init_function() { return true; }

int main(int argc, char* argv[])
{
    ScopeGuard scope_guard(argc, argv);
    return boost::unit_test::unit_test_main(&init_function, argc, argv);
}

