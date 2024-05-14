//===- KokkosTypes.td - Kokkos dialect types ------*- tablegen
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KOKKOS_TYPES_H
#define KOKKOS_TYPES_H

include "mlir/Dialect/Kokkos/IR/KokkosAttrDefs.td"
include "mlir/Dialect/Kokkos/IR/KokksoBase.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

// Base class for Builtin dialect types.
//class PartTensor_Type<string name, list<Trait> traits = [],
//                      string baseCppClass = "::mlir::Type">
//    : TypeDef<PartTensor_Dialect, name, traits, baseCppClass> {}

#endif
