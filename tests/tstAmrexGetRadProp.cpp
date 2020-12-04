#include <boost/test/unit_test.hpp>
#define BOOST_TEST_MODULE amrexgetradprop

#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <iostream>
#include <vector>

using namespace amrex;

AMREX_GPU_HOST_DEVICE
inline void initGasField(int i, int j, int k,
    amrex::Array4<amrex::Real> const& mf,
    amrex::Array4<amrex::Real> const& temp,
    amrex::Array4<amrex::Real> const& pressure, std::vector<amrex::Real> dx,
    std::vector<amrex::Real> plo, std::vector<amrex::Real> phi) noexcept
{
    constexpr int num_rad_species = 3;
    amrex::Real dTemp             = 5.0;
    amrex::Real dPres             = 0.05;
    amrex::Real x                 = plo[0] + (i + 0.5) * dx[0];
    amrex::Real y                 = plo[1] + (j + 0.5) * dx[1];
    amrex::Real pi                = 4.0 * std::atan(1.0);
    amrex::GpuArray<amrex::Real, 3> LL;
    amrex::GpuArray<amrex::Real, 3> PP;
    amrex::GpuArray<amrex::Real, 3> Y_lo;
    amrex::GpuArray<amrex::Real, 3> Y_hi;

    for (int n = 0; n < num_rad_species; n++)
    {
        LL[n] = phi[n] - plo[n];
        PP[n] = LL[n] / 4.0;
    }
    for (int n = 0; n < num_rad_species; n++)
    {
        Y_lo[n] = 0.0;
        Y_hi[n] = 0.3 / num_rad_species;
    }

    temp(i, j, k)     = 500.0 + dTemp * std::sin(2.0 * pi * y / PP[1]);
    pressure(i, j, k) = 1.0 + dPres * std::sin(2.0 * pi * y / PP[1]);

    for (int n = 0; n < num_rad_species; n++)
    {
        mf(i, j, k, n) = Y_lo[n] + (Y_hi[n] - Y_lo[n]) * x / LL[n];
    }
}

BOOST_AUTO_TEST_CASE(amrex_get_radprop)
{

    ParmParse pp;
    std::vector<int> npts { 128, 128, 128 };
    Box domain(IntVect(D_DECL(0, 0, 0)),
        IntVect(D_DECL(npts[0] - 1, npts[1] - 1, npts[2] - 1)));

    std::vector<Real> plo(3, 0), phi(3, 0), dx(3, 1);
    for (int i = 0; i < BL_SPACEDIM; ++i)
    {
        phi[i] = domain.length(i);
        dx[i]  = (phi[i] - plo[i]) / domain.length(i);
    }

    int max_size = 32;
    pp.query("max_size", max_size);
    BoxArray ba(domain);
    ba.maxSize(max_size);

    ParmParse ppa("amr");
    std::string pltfile("plt");
    ppa.query("plot_file", pltfile);

    DistributionMapping dm { ba };

    int num_grow                  = 0;
    constexpr int num_rad_species = 3;

    MultiFab mass_frac_rad(ba, dm, num_rad_species, num_grow);
    MultiFab temperature(ba, dm, 1, num_grow);
    MultiFab pressure(ba, dm, 1, num_grow);

    IntVect tilesize(D_DECL(10240, 8, 32));

    MultiFab absc(ba, dm, 1, num_grow);

    for (MFIter mfi(mass_frac_rad, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const Box& gbox = mfi.tilebox();

        Array4<Real> const& Yrad = mass_frac_rad.array(mfi);
        Array4<Real> const& T    = temperature.array(mfi);
        Array4<Real> const& P    = pressure.array(mfi);

        amrex::ParallelFor(
            gbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                initGasField(i, j, k, Yrad, T, P, dx, plo, phi);
                // getRadProp(tbx, Yrad, Y, P, absc);
            });
    }

    /*
        MultiFab VarPltInit(ba, dm, num_rad_species, num_grow);
        MultiFab::Copy(VarPltInit, mass_frac_rad, 0, 0, num_rad_species,
       num_grow); MultiFab::Copy(VarPltInit, temperature, 0, num_rad_species, 1,
       num_grow); MultiFab::Copy(VarPltInit, pressure, 0, num_rad_species, 1,
       num_grow); std::string initfile = amrex::Concatenate(pltfile, 99);
        PlotFileFromMF(VarPltInit, initfile);
    */
    BOOST_TEST(true);
}
