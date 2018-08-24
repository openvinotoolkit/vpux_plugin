#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/tensor/math.hpp"

mv::allocator mv::Tensor::allocator_;
mv::static_vector<mv::dim_type, mv::byte_type, mv::max_ndims> mv::Tensor::subsBuffer_;

mv::Tensor::Tensor(const string &name, const Shape &shape, DType dType, Order order) :
ComputationElement(name),
errValue(0.0f),
shape_(shape),
populated_(false)
{
    addAttr("dType", AttrType::DTypeType, dType);
    addAttr("order", AttrType::OrderType, order);
    addAttr("shape", AttrType::ShapeType, shape);
    addAttr("populated", AttrType::BoolType, false);
    logger_.log(Logger::MessageType::MessageDebug, "Defined tensor " + toString());
}

mv::Tensor::Tensor(mv::json::Value& v):
ComputationElement(v),
errValue(0.0f),
shape_(attributes_.at("shape").getContent<Shape>()),
populated_(attributes_.at("populated").getContent<bool>())
{
    if(populated_)
        data_ = std::make_shared<dynamic_vector<float_type>>(dynamic_vector<float_type>(shape_.totalSize()));
}

mv::Tensor::Tensor(const string &name, const Shape &shape, DType dType, Order order, const dynamic_vector<float_type>& data) :
Tensor(name, shape, dType, order)
{
    if (populate(data))
        getAttr("populated").setContent<bool>(true);
}

mv::Tensor::Tensor(const Tensor &other) :
ComputationElement(other),
shape_(other.shape_),
populated_(other.populated_)
{
    if (populated_)
        data_ = std::make_shared<dynamic_vector<float_type>>(dynamic_vector<float_type>(*other.data_));
    logger_.log(Logger::MessageType::MessageDebug, "Copied tensor " + toString());
}

mv::Tensor::Tensor() :
ComputationElement("unknown_tensor")
{
    logger_.log(Logger::MessageType::MessageWarning, "Defined unknown tensor");
    addAttr("dType", AttrType::DTypeType, DType::Unknown);
    addAttr("order", AttrType::OrderType, Order::Unknown);
    addAttr("shape", AttrType::ShapeType, Shape());
    addAttr("populated", AttrType::BoolType, false);
}

mv::Tensor::~Tensor()
{

}

bool mv::Tensor::populate(const dynamic_vector<float_type>& data, Order order)
{

    if (order != Order::Unknown && order != getOrder())
    {
        getAttr("order").setContent<Order>(order);
    }

    if (data.size() != getShape().totalSize())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to populate tensor - mismatch between input array size (" +
            Printable::toString((unsigned)data.size()) + ") and declared shape (" + getAttr("shape").getContentStr() + ")");
        return false;
    }

    data_ = std::make_shared<dynamic_vector<float>>(data);
    getAttr("populated").setContent<bool>(true);
    populated_ = true;
    return true;

}

bool mv::Tensor::unpopulate()
{
    if (!getAttr("populated").getContent<bool>())
        return false;

    data_.reset();
    getAttr("populated").setContent<bool>(false);
    populated_ = false;
    return true;

}

void mv::Tensor::reorder(Order order)
{

    Order oldOrder = getAttr("order").getContent<Order>();

    getAttr("order").setContent<Order>(order);

    if (!populated_)
        return;

    std::unique_ptr<OrderClass> oldOrderClass = mv::OrderFactory::createOrder(oldOrder);
    std::unique_ptr<OrderClass> newOrderClass = mv::OrderFactory::createOrder(order);

    auto dataPtr = std::make_shared<dynamic_vector<float>>(data_->size());

    for (unsigned i = 0; i < dataPtr->size(); ++i)
    {

        auto sub = oldOrderClass->indToSub(shape_, i);
        dataPtr->at(newOrderClass->subToInd(shape_, sub)) = data_->at(i);

    }

    data_ = dataPtr;

}

bool mv::Tensor::broadcast(const Shape& shape)
{

    if (!isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom element-wise operation using unpopulated tensor");
        return false;
    }

    Shape s1 = shape_, s2 = shape;

    if (s1.ndims() == 0 || s2.ndims() == 0)
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom element-wise operation using 0-dimensional tensor");
        return false;
    }

    if (s1 == s2)
        return true;
    else
    {

        Shape sO = Shape::broadcast(s1, s2);
        if (sO == Shape())
            return false;

        std::shared_ptr<dynamic_vector<float>> dataPtr = std::make_shared<dynamic_vector<float>>(sO.totalSize());

        if (s1.ndims() > s2.ndims())
        {
            s2 = Shape::augment(s2, s1.ndims());
        }
        else if (s2.ndims() > s1.ndims())
            s1 = Shape::augment(s1, s2.ndims());

        for (unsigned i = 0; i < dataPtr->size(); ++i)
        {

            static_vector<dim_type, byte_type, max_ndims> sub = indToSub_(sO, i);

            for (unsigned j = 0; j < sub.length(); ++j)
            {
                if (s1[j] == 1 && sO[j] > 1)
                    sub[j] = 0;
            }

            (*dataPtr)[i] = at(subToInd_(s1, sub));

        }

        shape_ = sO;
        data_ = dataPtr;
        return true;

    }

    return false;

}

// TODO - Handle the case when tensor got deleted, by the reference is still in use
mv::dynamic_vector<mv::float_type>& mv::Tensor::getData()
{
    if (!isPopulated())
        logger_.log(Logger::MessageType::MessageWarning, "Attempt of restoring data from an unpopulated tensor '" + name_ + "'");
    return *data_.get();
}

mv::DType mv::Tensor::getDType() const
{
    return getAttr("dType").getContent<DType>();
}

mv::Order mv::Tensor::getOrder() const
{
    return getAttr("order").getContent<Order>();
}

void mv::Tensor::setOrder(mv::Order order)
{
    getAttr("order").setContent<mv::Order>(order);
}

mv::string mv::Tensor::toString() const
{
    return "tensor '" + name_ + "' " + ComputationElement::toString();
}

bool mv::Tensor::elementWise_(const Tensor& other, const std::function<float(float, float)>& opFunc)
{

    if (!isPopulated() || !other.isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom element-wise operation using unpopulated tensor");
        return false;
    }

    Shape s1 = shape_, s2 = other.getShape();

    if (s1.ndims() == 0 || s2.ndims() == 0)
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom element-wise operation using 0-dimensional tensor");
        return false;
    }

    if (s1 == s2)
    {
        for (unsigned i = 0; i < data_->size(); ++i)
            (*data_)[i] = opFunc(at(i), other(i));
        return true;
    }
    else
    {

        /*Tensor broadcastOther(other);
        if (!broadcastOther.broadcast(s1))
            return false;
        if (!broadcast(s2))
            return false;

        std::transform(data_->begin(), data_->end(), broadcast.data_->begin(), data_->begin(), opFunc);
        return true;*/

        Shape sO = Shape::broadcast(s1, s2);
        std::shared_ptr<dynamic_vector<float>> dataPtr;

        if (sO == getShape())
        {
            dataPtr = data_;
        }
        else
        {
            dataPtr = std::make_shared<dynamic_vector<float>>(sO.totalSize());
        }

        if (s1.ndims() > s2.ndims())
        {
            s2 = Shape::augment(s2, s1.ndims());
        }
        else if (s2.ndims() > s1.ndims())
            s1 = Shape::augment(s1, s2.ndims());

        for (unsigned i = 0; i < dataPtr->size(); ++i)
        {

            static_vector<dim_type, byte_type, max_ndims> subO = indToSub_(sO, i);
            static_vector<dim_type, byte_type, max_ndims> sub1 = subO, sub2 = subO;

            for (unsigned j = 0; j < subO.length(); ++j)
            {
                if (s1[j] == 1 && sO[j] > 1)
                    sub1[j] = 0;
                if (s2[j] == 1 && sO[j] > 1)
                    sub2[j] = 0;
            }

            (*dataPtr)[i] = opFunc(at(subToInd_(s1, sub1)), other.at(subToInd_(s2, sub2)));

        }

        shape_ = sO;
        data_ = dataPtr;
        return true;

    }

    return false;
}

bool mv::Tensor::add(const Tensor& other)
{
    return elementWise_(other, std::plus<float>());
}

bool mv::Tensor::add(float val)
{

    if (!isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom scalar operation using unpopulated tensor");
        return false;
    }
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] += val;

    return true;

}

bool mv::Tensor::subtract(const Tensor& other)
{
    return elementWise_(other, std::minus<float>());
}

bool mv::Tensor::subtract(float val)
{

    if (!isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom scalar operation using unpopulated tensor");
        return false;
    }
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] -= val;

    return true;

}

bool mv::Tensor::multiply(const Tensor& other)
{
    return elementWise_(other, std::multiplies<float>());
}

bool mv::Tensor::multiply(float val)
{

    if (!isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom scalar operation using unpopulated tensor");
        return false;
    }
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] *= val;

    return true;

}

bool mv::Tensor::divide(float val)
{

    if (!isPopulated())
    {
        Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom scalar operation using unpopulated tensor");
        return false;
    }
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] /= val;

    return true;

}

bool mv::Tensor::divide(const Tensor& other)
{
    return elementWise_(other, std::divides<float>());
}

bool mv::Tensor::sqrt()
{
    for (unsigned i = 0; i < data_->size(); ++i)
        (*data_)[i] = std::sqrt((*data_)[i]);

    return true;
}

mv::Logger& mv::Tensor::logger()
{
    return logger_;
}

mv::float_type& mv::Tensor::at(const static_vector<dim_type, byte_type, max_ndims>& sub)
{
    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return (*data_)[subToInd(sub)];
}

const mv::float_type& mv::Tensor::at(const static_vector<dim_type, byte_type, max_ndims>& sub) const
{
    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return (*data_)[subToInd(sub)];
}

mv::float_type& mv::Tensor::at(unsigned idx)
{

    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return (*data_)[idx];

}

const mv::float_type& mv::Tensor::at(unsigned idx) const
{

    if (!isPopulated())
    {
        logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
        return errValue;
    }

    return (*data_)[idx];

}

mv::float_type& mv::Tensor::operator()(unsigned idx)
{
    return at(idx);
}

const mv::float_type& mv::Tensor::operator()(unsigned idx) const
{
    return at(idx);
}

mv::float_type& mv::Tensor::operator()(const static_vector<dim_type, byte_type, max_ndims>& sub)
{
    return at(sub);
}

const mv::float_type& mv::Tensor::operator()(const static_vector<dim_type, byte_type, max_ndims>& sub) const
{
    return at(sub);
}
