#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE ponesolveeb

#include <AMRParam.hpp>
#include <MLMGParam.hpp>
#include <POneEquationEB.hpp>

BOOST_AUTO_TEST_CASE(p1_solve_eb)
{
    amrex::ParmParse pp;
    PeleRad::AMRParam amrpp(pp);
    PeleRad::MLMGParam mlmgpp(pp);

    // initialize the grid
    int nlevels = amrpp.max_level_ + 1;
    Vector<amrex::Geometry> geom(nlevels);
    Vector<BoxArray> grids(nlevels);
    // Vector<amrex::DistributionMapping> dmap(nlevels);

    amrex::RealBox rb({ AMREX_D_DECL(0.0, 0.0, 0.0) }, { 1.0, 1.0, 1.0 });
    std::array<int, AMREX_SPACEDIM> isperiodic { AMREX_D_DECL(0, 0, 0) };
    amrex::Geometry::Setup(&rb, 0, isperiodic.data());
    Box domain0(IntVect { AMREX_D_DECL(0, 0, 0) },
        IntVect { AMREX_D_DECL(amrpp.n_cell_ - 1, amrpp.n_cell_ - 1, 1) });
    Box domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        geom[ilev].define(domain);
        domain.refine(amrpp.ref_ratio_);
    }

    grids[0].define(domain0);
    grids[0].maxSize(amrpp.max_grid_size_);

    PeleRad::POneEquationEB rte(amrpp, mlmgpp, geom, grids);
    rte.solve();
    rte.write();

    BOOST_TEST(true);
}
