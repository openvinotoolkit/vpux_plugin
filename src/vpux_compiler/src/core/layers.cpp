//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/layers.hpp"

using namespace vpux;

//
// Dim3D
//

const Dim vpux::Dims3D::Act::B(0);
const Dim vpux::Dims3D::Act::H(1);
const Dim vpux::Dims3D::Act::IC(2);

const Dim vpux::Dims3D::Filter::B(0);
const Dim vpux::Dims3D::Filter::IC(1);
const Dim vpux::Dims3D::Filter::OC(2);

const Dim vpux::Dims3D::Output::B(0);
const Dim vpux::Dims3D::Output::H(1);
const Dim vpux::Dims3D::Output::OC(1);

//
// Dims4D
//

const Dim vpux::Dims4D::Act::N(0);
const Dim vpux::Dims4D::Act::C(1);
const Dim vpux::Dims4D::Act::H(2);
const Dim vpux::Dims4D::Act::W(3);

const Dim vpux::Dims4D::Filter::OC(0);
const Dim vpux::Dims4D::Filter::IC(1);
const Dim vpux::Dims4D::Filter::KY(2);
const Dim vpux::Dims4D::Filter::KX(3);

const Dim vpux::Dims4D::Kernel::Y(0);
const Dim vpux::Dims4D::Kernel::X(1);

const Dim vpux::Dims4D::Dilation::Y(0);
const Dim vpux::Dims4D::Dilation::X(1);

const Dim vpux::Dims4D::Strides::Y(0);
const Dim vpux::Dims4D::Strides::X(1);

const Dim vpux::Dims4D::PadsBegin::Top(0);
const Dim vpux::Dims4D::PadsBegin::Left(1);

const Dim vpux::Dims4D::PadsEnd::Bottom(0);
const Dim vpux::Dims4D::PadsEnd::Right(1);

const Dim vpux::Dims4D::PadsOutput::Y(0);
const Dim vpux::Dims4D::PadsOutput::X(1);

//
// Dims5D
//

const Dim vpux::Dims5D::Act::N(0);
const Dim vpux::Dims5D::Act::C(1);
const Dim vpux::Dims5D::Act::D(2);
const Dim vpux::Dims5D::Act::H(3);
const Dim vpux::Dims5D::Act::W(4);

const Dim vpux::Dims5D::Filter::OC(0);
const Dim vpux::Dims5D::Filter::IC(1);
const Dim vpux::Dims5D::Filter::KZ(2);
const Dim vpux::Dims5D::Filter::KY(3);
const Dim vpux::Dims5D::Filter::KX(4);

const Dim vpux::Dims5D::Kernel::Z(0);
const Dim vpux::Dims5D::Kernel::Y(1);
const Dim vpux::Dims5D::Kernel::X(2);

const Dim vpux::Dims5D::Dilation::Z(0);
const Dim vpux::Dims5D::Dilation::Y(1);
const Dim vpux::Dims5D::Dilation::X(2);

const Dim vpux::Dims5D::Strides::Z(0);
const Dim vpux::Dims5D::Strides::Y(1);
const Dim vpux::Dims5D::Strides::X(2);

const Dim vpux::Dims5D::PadsBegin::Front(0);
const Dim vpux::Dims5D::PadsBegin::Top(1);
const Dim vpux::Dims5D::PadsBegin::Left(2);

const Dim vpux::Dims5D::PadsEnd::Back(0);
const Dim vpux::Dims5D::PadsEnd::Bottom(1);
const Dim vpux::Dims5D::PadsEnd::Right(2);

const Dim vpux::Dims5D::PadsOutput::Z(0);
const Dim vpux::Dims5D::PadsOutput::Y(1);
const Dim vpux::Dims5D::PadsOutput::X(2);

//
// DimsGroups5D
//

const Dim vpux::DimsGroups5D::Act::G(0);
const Dim vpux::DimsGroups5D::Act::N(1);
const Dim vpux::DimsGroups5D::Act::C(2);
const Dim vpux::DimsGroups5D::Act::H(3);
const Dim vpux::DimsGroups5D::Act::W(4);

const Dim vpux::DimsGroups5D::Filter::G(0);
const Dim vpux::DimsGroups5D::Filter::OC(1);
const Dim vpux::DimsGroups5D::Filter::IC(2);
const Dim vpux::DimsGroups5D::Filter::KY(3);
const Dim vpux::DimsGroups5D::Filter::KX(4);

const Dim vpux::DimsGroups5D::Kernel::Y(0);
const Dim vpux::DimsGroups5D::Kernel::X(1);

const Dim vpux::DimsGroups5D::Dilation::Y(0);
const Dim vpux::DimsGroups5D::Dilation::X(1);

const Dim vpux::DimsGroups5D::Strides::Y(0);
const Dim vpux::DimsGroups5D::Strides::X(1);

const Dim vpux::DimsGroups5D::PadsBegin::Top(0);
const Dim vpux::DimsGroups5D::PadsBegin::Left(1);

const Dim vpux::DimsGroups5D::PadsEnd::Bottom(0);
const Dim vpux::DimsGroups5D::PadsEnd::Right(1);

const Dim vpux::DimsGroups5D::PadsOutput::Y(0);
const Dim vpux::DimsGroups5D::PadsOutput::X(1);
