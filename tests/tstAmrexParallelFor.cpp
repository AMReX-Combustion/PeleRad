#include <boost/test/unit_test.hpp>
#define BOOST_TEST_MODULE amrexparallelfor

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

BOOST_AUTO_TEST_CASE(amrex_parallel_for)
{
    BoxArray ba;

    int n_cell        = 256;
    int max_grid_size = 64;
    ParmParse pp;
    pp.query("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);
    Box domain_box(IntVect(0), IntVect(n_cell - 1));
    ba.define(domain_box);
    ba.maxSize(max_grid_size);

    MultiFab mf(ba, DistributionMapping { ba }, 1, 0);
    mf.setVal(0.0);

    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx           = mfi.tilebox();
        Array4<Real> const& fab = mf.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) { fab(i, j, k) += 1.0; });
    }

    BOOST_TEST(true);
}

