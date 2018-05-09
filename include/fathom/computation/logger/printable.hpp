#ifndef PRINTABLE_HPP_
#define PRINTABLE_HPP_

#include <string>
#include "include/fathom/computation/model/types.hpp"

namespace mv
{

    class Printable
    {

    public:

        virtual string toString() const = 0;

        static string toString(const Printable &obj)
        {
            return obj.toString();
        }

        static string toString(int_type value)
        {
            return std::to_string(value);
        }

        static string toString(float_type value)
        {
            return std::to_string(value);
        }

        static string toString(unsigned_type value)
        {
            return std::to_string(value);
        }

        static string toString(byte_type value)
        {
            return toString((unsigned_type)value);
        }

        static string toString(dim_type value)
        {
            return toString((unsigned_type)value);
        }

        static string toString(DType value)
        {
            switch (value)
            {
                case DType::Float:
                    return "float";

                default:
                    return "unknown";
            }
        }

        static string toString(Order value)
        {
            switch (value)
            {
                case Order::NWHC:
                    return "NWHC";

                default:
                    return "unknown";
            }
        }

    };

}

#endif
