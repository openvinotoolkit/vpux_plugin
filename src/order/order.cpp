#include "include/mcm/order/order.hpp"

std::unordered_map<std::size_t, std::string> mv::Order::rowMajorID =
{
    {1, "W"},
    {2, "WH"},
    {3, "WHC"},
    {4, "WHCN"},
    {5, "WHCNT"}
};

std::unordered_map<std::size_t, std::string> mv::Order::colMajorID =
{
    {1, "W"},
    {2, "HW"},
    {3, "CHW"},
    {4, "NCHW"},
    {5, "TNCHW"}
};

std::size_t mv::Order::subToInd(const Shape &s, const std::vector<std::size_t>& sub) const
{

    //No 0-dimensional shapes
    if (s.ndims() == 0)
        throw ShapeError(*this, "subToInd: Cannot compute subscripts for 0-dimensional shape");

    //No shapes bigger than dimension supported
    if (s.ndims() != contVector_.size())
        throw ShapeError(*this, "subToInd: Mismatch between number of dimensions in shape ("
         + std::to_string(s.ndims()) + ") and dimensions supported by this mv::Order " + std::to_string(contVector_.size()));

    //If shape is correct, also sub has to be correct
    if (sub.size() != s.ndims())
        throw ShapeError(*this, "subToInd: Mismatch between subscript vector (length " + std::to_string(sub.size()) +
            ") and number of dimensions in shape (" + std::to_string(s.ndims()) + ")");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (int i = contVector_.size() - 1; i >= 0; --i)
    {

        if (sub[contVector_[i]] >=  s[contVector_[i]])
            throw ShapeError(*this, "subToInd: Subscript " + std::to_string(sub[contVector_[i]]) + " exceeds the dimension " +
                std::to_string(s[contVector_[i]]));

        currentResult += currentMul * sub[contVector_[i]];
        currentMul *= s[contVector_[i]];

    }

    return currentResult;

}

std::vector<std::size_t> mv::Order::indToSub(const Shape &s, std::size_t idx) const
{

    if (s.ndims() == 0)
        throw ShapeError(*this, "indToSub: Cannot compute subscripts for 0-dimensional shape");

    std::vector<std::size_t> sub(s.ndims());
    sub[contVector_[contVector_.size() - 1]] =  idx % s[contVector_.size() - 1];
    int offset = -sub[contVector_.size() - 1];
    int scale = s[contVector_.size() - 1];
    for (int i = contVector_.size() - 2; i >= 0; --i)
    {
        sub[contVector_[i]] = (idx + offset) / scale % s[contVector_[i]];
        offset -= sub[contVector_[i]] * scale;
        scale *= s[contVector_[i]];
    }

    return sub;

}

//Read only access to dimensions
std::size_t mv::Order::operator[](std::size_t idx) const
{
    return contVector_[idx];
}

bool mv::Order::operator!=(const mv::Order& other) const
{
    return contVector_ != other.contVector_;
}

bool mv::Order::operator==(const mv::Order& other) const
{
    return contVector_ == other.contVector_;
}

std::size_t mv::Order::size() const
{
    return contVector_.size();
}

mv::Order::Order(const mv::Order& other)
    :contVector_(other.contVector_)
{

}

mv::Order& mv::Order::operator=(const mv::Order& other)
{
    contVector_ = other.contVector_;
    return *this;
}

std::string mv::Order::toString() const
{
    std::string to_return("(");
    for(auto i: contVector_)
        to_return += std::to_string(i) + " ";
    to_return += ")";
    return to_return;
}

bool mv::Order::isRowMajor(const mv::Order& o)
{
    if(o == mv::Order("W"))
        return true;
    if(o == mv::Order("WH"))
        return true;
    if(o == mv::Order("WHC"))
        return true;
    if(o == mv::Order("WHCN"))
        return true;
    if(o == mv::Order("WHCNT"))
        return true;
    return false;
}



bool mv::Order::isColMajor(const mv::Order& o)
{
    if(o == mv::Order("W"))
        return true;
    if(o == mv::Order("WH"))
        return true;
    if(o == mv::Order("WHC"))
        return true;
    if(o == mv::Order("WHCN"))
        return true;
    if(o == mv::Order("WHCNT"))
        return true;
    return false;
}

bool mv::Order::isRowMajorPlanar(const mv::Order& o)
{
    if(o == mv::Order("W"))
        return true;
    if(o == mv::Order("WH"))
        return true;
    if(o == mv::Order("WHC"))
        return true;
    if(o == mv::Order("WHCN"))
        return true;
    if(o == mv::Order("WHCNT"))
        return true;
    return false;
}

bool mv::Order::isColMajorPlanar(const mv::Order& o)
{
    if(o == mv::Order("W"))
        return true;
    if(o == mv::Order("WH"))
        return true;
    if(o == mv::Order("WHC"))
        return true;
    if(o == mv::Order("WHCN"))
        return true;
    if(o == mv::Order("WHCNT"))
        return true;
    return false;
}

bool mv::Order::isRowInterleaved(const mv::Order& o)
{
    if(o == mv::Order("W"))
        return true;
    if(o == mv::Order("WH"))
        return true;
    if(o == mv::Order("WHC"))
        return true;
    if(o == mv::Order("WHCN"))
        return true;
    if(o == mv::Order("WHCNT"))
        return true;
    return false;
}

std::string mv::Order::getLogID() const
{
    return "Order '" + toString() + "'";
}

