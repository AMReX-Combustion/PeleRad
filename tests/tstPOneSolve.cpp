#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE ponesolve

#include <AMRParam.hpp>
#include <MLMGParam.hpp>
#include <POneEquation.hpp>

BOOST_AUTO_TEST_CASE(p1_solve)
{
    amrex::ParmParse pp;
    PeleRad::AMRParam amrpp(pp);
    PeleRad::MLMGParam mlmgpp(pp);
    PeleRad::POneEquation rte(amrpp, mlmgpp);
    rte.solve();
    // rte.write();

    BOOST_TEST(true);
}
