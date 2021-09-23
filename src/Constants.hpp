#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace PeleRad
{

struct RadComps
{
    int co2Indx = -1;
    int h2oIndx = -1;
    int coIndx = -1;
    
    void checkIndices()
    {
        if(co2Indx > 0) std::cout << " include radiative gas: CO2, the index is "<< co2Indx << std::endl;
        if(h2oIndx > 0) std::cout << " include radiative gas: H2O, the index is "<< h2oIndx << std::endl;
        if(coIndx > 0) std::cout << " include radiative gas: CO, the index is " << coIndx << std::endl;
    }
};

}
#endif
