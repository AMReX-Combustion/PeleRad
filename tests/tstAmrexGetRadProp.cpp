#include <boost/test/unit_test.hpp>
#define BOOST_TEST_MODULE amrexgetradprop

#include <PlanckMean.hpp>
#include <SpectralModels.hpp>

#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <iostream>
#include <vector>

using namespace amrex;

AMREX_GPU_HOST_DEVICE
inline void initGasField(int i, int j, int k, const Array4<Real>& mf,
    const Array4<Real>& temp, const Array4<Real>& pressure,
    const GpuArray<Real, 3ul>& dx, const GpuArray<Real, 3ul>& plo,
    const GpuArray<Real, 3ul>& phi) noexcept
{
    constexpr int num_rad_species = 3;
    constexpr Real dTemp          = 5.0;
    constexpr Real dPres          = 0.05;
    Real x                        = plo[0] + (i + 0.5) * dx[0];
    Real y                        = plo[1] + (j + 0.5) * dx[1];
    Real pi                       = 4.0 * std::atan(1.0);
    GpuArray<Real, 3> LL;
    GpuArray<Real, 3> PP;
    GpuArray<Real, 3> Y_lo;
    GpuArray<Real, 3> Y_hi;

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
    std::string data_path;

#ifdef DATABASE_PATH
    data_path = DATABASE_PATH;
#else
    data_path = "../../data/kpDB/";
#endif

    using PeleRad::PlanckMean;
    using PeleRad::RadProp::getRadProp;

    PlanckMean radprop(data_path);

    ParmParse pp;
    std::vector<int> npts { 128, 128, 128 };
    Box domain(IntVect(D_DECL(0, 0, 0)),
        IntVect(D_DECL(npts[0] - 1, npts[1] - 1, npts[2] - 1)));

    GpuArray<Real, 3ul> plo;
    GpuArray<Real, 3ul> phi;
    GpuArray<Real, 3ul> dx;

    for (int i = 0; i < AMREX_SPACEDIM; ++i)
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

    const auto& kpco2 = radprop.kpco2();
    const auto& kph2o = radprop.kph2o();
    const auto& kpco  = radprop.kpco();

    MultiFab absc(ba, dm, 1, num_grow);

    for (MFIter mfi(mass_frac_rad, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const Box& gbox = mfi.tilebox();

        const Array4<Real>& Yrad  = mass_frac_rad.array(mfi);
        const Array4<Real>& T     = temperature.array(mfi);
        const Array4<Real>& P     = pressure.array(mfi);
        const Array4<Real>& kappa = absc.array(mfi);

        ParallelFor(gbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            initGasField(i, j, k, Yrad, T, P, dx, plo, phi);
            getRadProp(i, j, k, Yrad, T, P, kappa, kpco2, kph2o, kpco);
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
