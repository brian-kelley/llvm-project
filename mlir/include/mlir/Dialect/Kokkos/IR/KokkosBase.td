//===- KokkosBase.td - Kokkos dialect base ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KOKKOS_BASE
#define KOKKOS_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def Kokkos_Dialect : Dialect {
  let name = "kokkos";
  let cppNamespace = "::mlir::kokkos";
  let description = [{
     The `kokkos` dialect provides MLIR representations for Kokkos
     constructs like hierarchical parallelism, parallel_scan, Kokkos
     memory spaces and execution space instances.
  }];
}

// The predicate indicates the type of the comparison to perform:
// (un)orderedness, (in)equality and less/greater than (or equal to) as
// well as predicates that are always true or false.
def Kokkos_MemorySpace : I64EnumAttr<
    "MemorySpace", "",
    [
      I64EnumAttrCase<"Host", 0, "host">,
      I64EnumAttrCase<"Device", 1, "device">,
      I64EnumAttrCase<"DualView", 2, "dualview">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

def Kokkos_ExecutionSpace : I64EnumAttr<
    "ExecutionSpace", "",
    [
      I64EnumAttrCase<"Host", 0, "host">,
      I64EnumAttrCase<"Device", 1, "device">,
      I64EnumAttrCase<"TeamHandle", 2, "team_handle">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

def Kokkos_ParallelLevel : I64EnumAttr<
    "ParallelLevel", "",
    [
      I64EnumAttrCase<"RangePolicy", 0, "range">,
      I64EnumAttrCase<"TeamPolicy", 1, "team">,
      I64EnumAttrCase<"TeamVector", 2, "team_vector">,
      I64EnumAttrCase<"TeamThread", 3, "team_thread">,
      I64EnumAttrCase<"ThreadVector", 4, "thread_vector">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

def Kokkos_SingleLevel : I64EnumAttr<
    "SingleLevel", "",
    [
      I64EnumAttrCase<"PerTeam", 0, "per_team">,
      I64EnumAttrCase<"PerThread", 1, "per_thread">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

#endif
