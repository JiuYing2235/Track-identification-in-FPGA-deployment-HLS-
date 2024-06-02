#include "param.h"
#include <iostream>
#include <iomanip>
#include "ap_fixed.h"
#include <cstdint>
#include <typeinfo> // Include for typeid


int main() {
    f_32 img_in[3][128][128] = {
          #include "img_data_1/images (4).h"
    };
    f_32 Cout[2];


    resnet(img_in, Cout);


    std::cout << "out[Defective] = " << Cout[0] << std::endl;
    std::cout << "out[Non defective] = " << Cout[1] << std::endl;
}


