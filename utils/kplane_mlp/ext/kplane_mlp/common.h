#ifndef KPLANE_MLP_COMMON_H
#define KPLANE_MLP_COMMON_H

enum FeatFusion : unsigned char { SUM=0, AVG=1, MUL=2 };
enum Activation : unsigned char { Softplus=0, ReLU=1, None=255};

template <Activation act>
struct is_softplus{static const bool value{false};};

template <>
struct is_softplus<Activation::Softplus>{static const bool value{false};};


template <Activation act>
struct is_relu{static const bool value{false};};

template <>
struct is_relu<Activation::ReLU>{static const bool value{false};};


#endif