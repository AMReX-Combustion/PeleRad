#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <fstream>

#define BOOST_TEST_MODULE boostreaddatabase
#include <AMReX_MultiFab.H>

namespace but = boost::unit_test;
namespace fs  = std::filesystem;

BOOST_AUTO_TEST_CASE(boost_read_database, *but::tolerance(0.00001))
{
    std::string data_path;

#ifdef DATABASE_PATH
    data_path = DATABASE_PATH;
#else
    data_path = "../../data/kpDB/";
#endif

    fs::path kplco2(data_path + "kpl_co2.dat");

    BOOST_TEST(fs::exists(kplco2));

    std::ifstream data(kplco2);

    amrex::GpuArray<amrex::Real, 126ul> T;
    amrex::GpuArray<amrex::Real, 126ul> kpco2;

    size_t i = 0;

    for (float T_temp, kp_temp; data >> T_temp >> kp_temp;)
    {
        // std::cout << "Reading from file:" << T_temp << " " << kp_temp
        //          << " i=" << i << '\n';
        T[i]     = T_temp;
        kpco2[i] = kp_temp;
        i++;
    }

    BOOST_TEST(T[15] == 600.);
    BOOST_TEST(kpco2[15] == 0.33684464);
    BOOST_TEST(T[107] == 2440.);
    BOOST_TEST(kpco2[107] == 0.07427838);

    data.close();
}

