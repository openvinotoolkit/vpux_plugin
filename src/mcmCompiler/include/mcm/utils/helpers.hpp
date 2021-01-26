#include <fstream>

namespace mv {
namespace utils {

template <typename T, void (*F)(T*)>
struct RaiiWrapper {
    void operator()(T* obj_ptr) {
        if(obj_ptr) {
            F(obj_ptr);
        }
    }
};

void releaseFile(FILE* ptr);

}  // namespace utils
}  // namespace mv
