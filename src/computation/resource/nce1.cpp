#include "mcm/algorithms/dijkstra.hpp"
#include "mcm/computation/resource/nce1.hpp"
#include "mcm/utils/custom_math.hpp"
#include <cmath>

mv::Nce1::Nce1()
    :nce1_dpe(256),
     input_data_size(2),
     coefficients_storage_dimension(pow(2,17)), //128K
     data_storage_dimension(pow(2,17)), //128K
     cmx_stream_size(pow(2,18)), //256K
     max_coefficient_number_per_line(256),
     max_descriptors_x_hw_op(255),
     split_by_input_channel_overhead(30000),
     split_by_width_overhead(5000),
     split_by_output_channel_overhead(7000),
     bits_per_line(128)
{
    dpe_x_output_channel =
    {
        {0, 1},
        {1, 2},
        {2, 4},
        {3, 8},
        {4, 16}
    };

    output_channel_performed_one_shot =
    {
        {0, 256},
        {1, 128},
        {2, 64},
        {3, 32},
        {4, 16}
    };

    reverse_output_channel_performed_one_shot =
    {
        {256, 0},
        {128, 1},
        {64, 2},
        {32, 3},
        {16, 4}
    };
}

mv::ModeSelectionResult mv::Nce1::optimize_convolution(const ModeSelectionNode source)
{
    mv::ModeSelectionNode target;
    target.remaining_output_channels = 0;

    auto generate_neighbours_lambda = [this](ModeSelectionNode a) {return generateNeighboursComingFromValidModes(a);};
    auto computer_cost_lambda = [this](ModeSelectionNode a, ModeSelectionNode b) {return computeModeCost(a, b);};

    return mv::dijkstraRT<mv::ModeSelectionNode, mv::ModeSelectionDistance>(source, target, generate_neighbours_lambda, computer_cost_lambda);
}

//At least one channel per ramblock
unsigned mv::Nce1::get_max_mode(unsigned input_channels)
{
    unsigned mode_to_return = mv::Mode0;
    for(; mode_to_return < mv::Mode4; ++mode_to_return)
        if(dpe_x_output_channel.at(mode_to_return) >= input_channels)
            break;
    return mode_to_return;
}

std::vector<unsigned> mv::Nce1::get_valid_modes(mv::ModeSelectionNode node)
{
    std::vector<unsigned> to_return;
    unsigned max_mode = get_max_mode(node.parameters.input_channels);
    for(unsigned mode = mv::Mode0; mode <= mv::Mode4; ++mode)
    {
        if(mode > max_mode)
            continue;
        unsigned number_of_descriptors_needed = ceil((double)(node.remaining_output_channels) / output_channel_performed_one_shot.at(mode));
        if(number_of_descriptors_needed >= max_descriptors_x_hw_op) //Useless check with the current numbers, but you never know
            continue;
        to_return.push_back(mode);
    }
    return to_return;
}

std::vector<mv::ModeSelectionNode> mv::Nce1::generateNeighboursComingFromValidModes(const ModeSelectionNode current_node)
{
    std::set<ModeSelectionNode> valid_neighbours_set;
    std::vector<unsigned> valid_modes = get_valid_modes(current_node);
    unsigned n = valid_modes.size();
    for(unsigned i = 0; i < n; ++i)
    {
        mv::ModeSelectionNode neighbour(current_node);
        neighbour.remaining_output_channels -= output_channel_performed_one_shot.at(valid_modes[i]);
        if(neighbour.remaining_output_channels < 0)
            neighbour.remaining_output_channels = 0;
        valid_neighbours_set.insert(neighbour);
    }
    return std::vector<mv::ModeSelectionNode>(valid_neighbours_set.begin(), valid_neighbours_set.end());
}

bool mv::Nce1::check_min_lines_constraint(unsigned kernel_y, unsigned stride_y, unsigned input_width, unsigned input_channels)
{
    unsigned min_lines = kernel_y + stride_y + 2;
    unsigned space_required = min_lines * input_width * input_data_size * input_channels;
    return space_required > data_storage_dimension;
}

bool mv::Nce1::check_min_lines_constraint_(mv::ConvolutionParameters param)
{
    return check_min_lines_constraint(param.kernel_y, param.stride_y, param.input_width, param.input_channels);
}

bool mv::Nce1::check_coefficient_size_constraint(unsigned kernel_x, unsigned kernel_y, unsigned input_channels, unsigned output_channel_performed)
{
    unsigned coeff_size = kernel_x * kernel_y * input_channels * getActualOutputChannels(output_channel_performed) * input_data_size;
    return coeff_size > coefficients_storage_dimension;
}

bool mv::Nce1::check_coefficient_size_constraint_(mv::ConvolutionParameters param, unsigned output_channel_performed)
{
    return check_coefficient_size_constraint(param.kernel_x, param.kernel_y, param.input_channels, output_channel_performed);
}

bool mv::Nce1::check_coefficient_line_constraint(unsigned input_channels, unsigned kernel_x, unsigned kernel_y, int mode)
{
   unsigned channel_per_ramblock = input_channels / dpe_x_output_channel.at(mode);
   unsigned check_coeff_line_per_block = kernel_x * kernel_y * channel_per_ramblock;
   return check_coeff_line_per_block > max_coefficient_number_per_line;
}

bool mv::Nce1::check_coefficient_line_constraint_(mv::ConvolutionParameters param, int mode)
{
   return check_coefficient_line_constraint(param.input_channels, param.kernel_x, param.kernel_y, mode);
}

bool mv::Nce1::check_channels_per_ram_block(unsigned input_channels, int mode)
{
    return dpe_x_output_channel.at(mode) > input_channels;
}

bool mv::Nce1::check_channels_per_ram_block_(mv::ConvolutionParameters param, int mode)
{
    return check_channels_per_ram_block(param.input_channels, mode);
}

mv::ModeSelectionDistance mv::Nce1::split_by_input_channel(mv::ConvolutionParameters param, unsigned actual_output_channels, int mode, bool support_split_over_c)
{
    mv::ModeSelectionDistance to_return;

    // Min lines  = Kernel_height + kernel stride + 2
    unsigned min_lines = param.kernel_y + param.stride_y + 2;
    // maximum ic that I can process without conflicting with min line constraint
    unsigned max_ic_minlines = floor((double)(data_storage_dimension)/(min_lines * input_data_size * param.input_width));
    // maximum ic that I can process without conflicting with coefficient line per block constraint
    unsigned max_ic_ramblock = floor((double)(nce1_dpe)/(param.kernel_x*param.kernel_y)*dpe_x_output_channel.at(mode));

    // calculate the max input channels that can be processed without running out of memory:
    unsigned max_ic = std::min(max_ic_minlines, max_ic_ramblock);
    // calculate ramblock for the selected mode
    unsigned ramBlocks = dpe_x_output_channel.at(mode);
    while (max_ic > 0)
    {
        // max_ic should be divisible by ramblocks and the split should be integer
        if((param.input_channels % max_ic) || (max_ic % ramBlocks))
            max_ic--;
        else
            break;
    }
    if(max_ic == 0)
    {
        //This mode does not allow splits
        to_return.cost = -1;
        to_return.split_performed = mv::Splits::InputChannel;
        to_return.num_splits = 0;
        to_return.mode = mode;
    }

    // n of input split required (padded to next pow of 2: WHY?)
    unsigned n_split_c = getActualInputChannelSplits(param.input_channels/max_ic);
    unsigned actual_ic_per_split = int(ceil((double)(param.input_channels)/n_split_c));

    if((n_split_c < 2) || (actual_ic_per_split % ramBlocks) || (!support_split_over_c))
    {
        //This mode does not allow splits
        to_return.cost = -1;
        to_return.split_performed = mv::Splits::InputChannel;
        to_return.num_splits = 0;
        to_return.mode = mode;
    }

    ConvolutionParameters new_param(param);
    param.input_channels = actual_ic_per_split;

    if (check_coefficient_size_constraint_(new_param, mode) ||
        check_channels_per_ram_block_(new_param, mode)  ||
        check_coefficient_line_constraint_(new_param, mode))
    {
        //This mode does not allow splits
        to_return.cost = -1;
        to_return.split_performed = mv::Splits::InputChannel;
        to_return.num_splits = n_split_c;
        to_return.mode = mode;
    }

    unsigned cost = 0;
    for(unsigned i = 0; i < n_split_c; ++i)
    {
        // calculate the operation cost
        unsigned ic = std::min(actual_ic_per_split, param.input_channels-i*actual_ic_per_split);
        cost += param.kernel_x * param.kernel_y * param.output_width * param.output_height * ic / dpe_x_output_channel.at(mode);
    }
    // add the cost of summation over input channels
    cost += (n_split_c - 1) * (actual_output_channels * param.output_width*param.output_height + split_by_input_channel_overhead);
    to_return.cost = cost;
    to_return.split_performed = mv::Splits::InputChannel;
    to_return.num_splits = n_split_c;
    to_return.mode = mode;
    return to_return;
}

mv::ModeSelectionDistance mv::Nce1::split_by_width(mv::ConvolutionParameters param, int mode, bool support_split_over_w)
{
    ModeSelectionDistance to_return;
    unsigned min_lines = param.kernel_y + param.stride_y + 2;
    // Space required by min lines = min_lines * input_width * input data type size * num input channels
    unsigned space_required = min_lines * input_data_size * param.input_width * param.input_channels;
    unsigned n_split_w = int(ceil(double(space_required)/data_storage_dimension));

    unsigned split_output_width =  param.output_width/n_split_w;
    unsigned split_input_width =  param.input_width/n_split_w;

    if (check_channels_per_ram_block_(param, mode) || !support_split_over_w)
    {
        to_return.cost = -1;
        to_return.mode = mode;
        to_return.split_performed = mv::Splits::Width;
        to_return.num_splits = n_split_w;
        return to_return;
    }

    unsigned cost = 0;
    for(unsigned i = 0; i < n_split_w; ++i)
    {
        unsigned os_w = std::min(split_output_width, param.output_width-i*split_output_width);
        unsigned is_w = std::min(split_input_width, param.input_width-i*split_input_width);

        ConvolutionParameters new_param(param);
        new_param.output_width = os_w;
        new_param.input_width = is_w;
        if(check_coefficient_size_constraint_(new_param, mode) || check_coefficient_line_constraint_(new_param, mode))
        {
            to_return.cost = -1;
            to_return.mode = mode;
            to_return.split_performed = mv::Splits::Width;
            to_return.num_splits = n_split_w;
            return to_return;
        }

        cost += param.kernel_x * param.kernel_y * param.output_width * os_w * param.input_channels / dpe_x_output_channel.at(mode) + split_by_width_overhead;
    }
    to_return.cost = cost;
    to_return.num_splits = n_split_w;
    to_return.split_performed = mv::Splits::Width;
    to_return.mode = mode;

    return to_return;
}

mv::ModeSelectionDistance mv::Nce1::computeModeCost(const mv::ModeSelectionNode a, const mv::ModeSelectionNode b)
{
    mv::ModeSelectionDistance to_return;

    int split_over_output_channel_overhead = 0;
    int mode = 0;
    unsigned output_channel_performed = 0;

    if(b.remaining_output_channels == 0)
    {
        output_channel_performed = a.remaining_output_channels;
        std::vector<unsigned> valid_modes = get_valid_modes(a);
        for(mode = valid_modes[valid_modes.size()-1]; mode >= 0; --mode)
            if(output_channel_performed_one_shot.at(mode) >= output_channel_performed)
                break;
    }
    else
    {
        output_channel_performed = a.remaining_output_channels - b.remaining_output_channels;
        mode = reverse_output_channel_performed_one_shot.at(output_channel_performed);
    }

    if((unsigned)a.remaining_output_channels == a.parameters.output_channels) //a is source, no overhead in this case
        split_over_output_channel_overhead = 0;
    else
        split_over_output_channel_overhead = 7000;

    //Aligning parameters to current mode
    mv::ConvolutionParameters parameters = a.parameters;
    parameters.input_channels = getActualInputChannels(parameters.input_channels, mode);

    //These two actually can be done also outside of the function since they do not depend on the mode. But for now they are keeping them here.`
    parameters.output_channels = getActualOutputChannels(parameters.output_channels);
    parameters.input_width = getActualInputWidth(parameters.input_width);

    bool need_split_by_width_or_input_channel = check_min_lines_constraint_(parameters);
    bool need_split_by_input_channel = check_coefficient_size_constraint_(parameters, output_channel_performed) | check_coefficient_line_constraint_(parameters, mode);

    if(need_split_by_width_or_input_channel)
    {
        ModeSelectionDistance splitted_by_input_channel = split_by_input_channel(parameters, b.remaining_output_channels, mode);
        ModeSelectionDistance splitted_by_width = split_by_width(parameters, mode);
        if(splitted_by_input_channel < splitted_by_width)
            to_return = splitted_by_input_channel;
        else
            to_return = splitted_by_width;
    }
    else if(need_split_by_input_channel)
        to_return = split_by_input_channel(parameters, b.remaining_output_channels, mode);
    else
    {
        to_return.cost = parameters.kernel_x * parameters.kernel_y * parameters.input_width * parameters.input_height * parameters.input_channels / dpe_x_output_channel.at(mode);
        to_return.split_performed = Splits::NoSplit;
        to_return.num_splits = 1;
        if(check_channels_per_ram_block_(parameters, mode))
            to_return.cost = -1; //infinite cost
    }

    to_return.cost += split_over_output_channel_overhead;
    to_return.mode = mode;
    to_return.performed_output_channels = output_channel_performed;
    return to_return;
}

unsigned mv::Nce1::getSplitsOverH(unsigned total_memory_occupied_by_tensors)
{
    return (unsigned int) ceil(double(total_memory_occupied_by_tensors)/cmx_stream_size);
}


//Padding functions
unsigned mv::Nce1::getActualInputChannels(unsigned input_channels, unsigned mode)
{
    return mv::round_up(input_channels, dpe_x_output_channel.at(mode));
}

//Most lazy and safe overload ever.
unsigned mv::Nce1::getActualInputChannels(unsigned input_channels)
{
    return mv::round_up(input_channels, 16);
}

unsigned mv::Nce1::getActualOutputChannels(unsigned output_channels)
{
    return mv::round_up(output_channels, 8);
}

unsigned mv::Nce1::getActualInputWidth(unsigned input_width)
{
     return mv::round_up(input_width, 8);
}

unsigned mv::Nce1::getActualInputHeight(unsigned input_height)
{
    return input_height;
}

unsigned mv::Nce1::getActualInputChannelSplits(unsigned splits)
{
     return mv::next_greater_power_of_2(splits);
}

unsigned mv::Nce1::getActualOutputWidth(unsigned output_width)
{
    return output_width;
}

unsigned mv::Nce1::getActualOutputHeight(unsigned output_height)
{
    return output_height;
}

unsigned mv::Nce1::getBytesPerLine()
{
    return 128 / 8; // Rams are organized in rows of 128bits, courtesy of the docs
}

unsigned mv::Nce1::getWordsPerLine()
{
    return getBytesPerLine() / input_data_size;
}

unsigned mv::Nce1::computeLocalLineStride(unsigned input_width)
{
    unsigned pixels_per_input_line_rounded_up = mv::round_up(input_width, 8);
    return pixels_per_input_line_rounded_up / 8; // equation courtesy of the docs
}

unsigned mv::Nce1::computeDescriptorSplits(unsigned splits_over_height, unsigned splits_over_input_channels, float actual_output_channels, std::vector<unsigned>& modes)
{
    unsigned to_return = splits_over_height * splits_over_input_channels;
    unsigned n = modes.size();
    unsigned sum = 0;
    for(unsigned i = 0; i < n; ++i)
    {
        unsigned output_channel_performed = output_channel_performed_one_shot.at(modes[i]);
        sum += ceil(actual_output_channels / output_channel_performed);
        actual_output_channels -= output_channel_performed;
    }
    to_return *= sum;
    return to_return;
}

unsigned mv::Nce1::computeInputChannelsPerRamBlock(unsigned input_channels, unsigned mode)
{
    int ram_blocks = dpe_x_output_channel.at(mode);
    return input_channels / ram_blocks;
}

unsigned mv::Nce1::getMaxNumberOfLinesInDataStorage()
{
    return data_storage_dimension / getBytesPerLine(); //both quantities are in bytes, the result is adimensional (# of lines)
}

unsigned mv::Nce1::computeLinesPerChannel(unsigned input_channels, unsigned local_line_stride, unsigned mode)
{
    unsigned max_lines_in_data_storage = getMaxNumberOfLinesInDataStorage();
    unsigned max_lines_per_ram_block = max_lines_in_data_storage / dpe_x_output_channel.at(mode);

    unsigned channels_per_ram_block = computeInputChannelsPerRamBlock(input_channels, mode);
    return max_lines_per_ram_block / (local_line_stride * channels_per_ram_block);

}
