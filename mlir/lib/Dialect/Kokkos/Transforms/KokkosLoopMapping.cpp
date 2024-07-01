//===- KokkosLoopMapping.cpp - Pattern for kokkos-loop-mapping pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace {

// Get the parallel nesting depth of the given Op
// - If Op itself is a kokkos.parallel or scf.parallel, then that counts as 1
// - Otherwise, Op counts for 0
// - Each enclosing parallel counts for 1 more
int getOpParallelDepth(Operation* op)
{
  int depth = 0;
  if(isa<scf::ParallelOp>(op) || isa<kokkos::ParallelOp>(op))
    depth++;
  Operation* parent = op->getParentOp();
  if(parent)
    return depth + getOpParallelDepth(parent);
  // op has no parent
  return depth;
}

// Get the number of parallel nesting levels for the given ParallelOp
// - The op itself counts as 1
// - Each additional nesting level counts as another
int getParallelNumLevels(scf::ParallelOp op)
{
  int depth = 1;
  op->walk([&](scf::ParallelOp child) {
    int childDepth = getOpParallelDepth(child);
    if(childDepth > depth)
      depth = childDepth;
  });
  return depth;
}

// Rewrite the given scf.parallel as a kokkos.parallel, with the given execution space and nesting level
// (not for TeamPolicy loops)
LogicalResult scfParallelToKokkos(RewriterBase& rewriter, scf::ParallelOp op, kokkos::ExecutionSpace exec, kokkos::ParallelLevel level)
{
  auto bodyBuilder = 
  [&](OpBuilder& builder, Location loc, ValueRange newInductionVars)
  {
    // Use an IRMap to easily replace old induction vars with new in the new loop
    IRMapping irMap;
    for(auto it : enumerate(newInductionVars))
    {
      int idx = std::get<0>(it);
      irMap.map(op.getInductionVars()[idx], std::get<1>(it));
    }
    // Now clone all the ops in the old loop body into the new one
    Block& oldBlock = op.getBody().front();
    for(Operation* op : oldBlock.getOps())
    {
      // For ops with side effects, wrap in a Kokkos::single if level == TeamThread.
      // For the other 3 possible levels, this body is always exected exactly once so no single needed.
      // "Has side effects" doesn't correspond to an op trait so we have to handle on a case-by-case basis here.
      bool opHasSideEffects = isa<memref::StoreOp>(op) || isa<memref::AtomicRMWOp>(op);
      if(opHasSideEffects && level == kokkos::ParallelLevel::TeamThread) {
        auto single = builder.create<kokkos::SingleOp>(op->getLoc(), kokkos::SingleLevel::PerThread);
        auto singleBody = builder.createBlock(single.getRegion());
        auto savedIP = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(singleBody);
        builder.clone(*op, irMap);
        builder.restoreInsertionPoint(savedIP);
      }
      else {
        // Just clone the op into the block directly
        builder.clone(*op, irMap);
      }
    }
  };
  auto newOp = rewriter.create<kokkos::ParallelOp>(
    op.getLoc(), exec, kokkos::ParallelLevel::RangePolicy, op.getUpperBound(), op.getInitVals(), bodyBuilder);
}

LogicalResult scfParallelToKokkosTeam(RewriterBase& rewriter, scf::ParallelOp op, Value leagueSize, Value teamSize, Value vectorLength)
{
  auto bodyBuilder = 
  [&](OpBuilder& builder, Location loc, ValueRange newInductionVars)
  {
    // Use an IRMap to easily replace old induction vars with new in the new loop
    IRMapping irMap;
    for(auto it : enumerate(newInductionVars))
    {
      int idx = std::get<0>(it);
      irMap.map(op.getInductionVars()[idx], std::get<1>(it));
    }
    // Now clone all the ops in the old loop body into the new one
    Block& oldBlock = op.getBody().front();
    for(Operation* op : oldBlock.getOps())
    {
      // For ops with side effects, wrap in a Kokkos::single if level == TeamThread.
      // For the other 3 possible levels, this body is always exected exactly once so no single needed.
      // "Has side effects" doesn't correspond to an op trait so we have to handle on a case-by-case basis here.
      bool opHasSideEffects = isa<memref::StoreOp>(op) || isa<memref::AtomicRMWOp>(op);
      if(opHasSideEffects && level == kokkos::ParallelLevel::TeamThread) {
        auto single = builder.create<kokkos::SingleOp>(op->getLoc(), kokkos::SingleLevel::PerThread);
        auto singleBody = builder.createBlock(single.getRegion());
        auto savedIP = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(singleBody);
        builder.clone(*op, irMap);
        builder.restoreInsertionPoint(savedIP);
      }
      else {
        // Just clone the op into the block directly
        builder.clone(*op, irMap);
      }
    }
  };
  auto newOp = rewriter.create<kokkos::ParallelOp>(
    op.getLoc(), exec, kokkos::ParallelLevel::RangePolicy, op.getUpperBound(), op.getInitVals(), bodyBuilder);
  rewriter.replaceOp(op, newOp);
  return success();
}

LogicalResult scfParallelToSequential(RewriterBase& rewriter, scf::ParallelOp op)
{
}

struct KokkosLoopRewriter : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  KokkosLoopRewriter(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
    // Only match with top-level ParallelOps (meaning op is not enclosed in another ParallelOp)
    if(op->getParentOfType<scf::ParallelOp>())
      return failure();
    // Determine the maximum depth of parallel nesting (a simple RangePolicy is 1, etc.)
    int nestingLevel = getParallelNumLevels(op);
    // Now decide whether this op should execute on device (offloaded) or host.
    // Operations that are assumed to be host-only:
    // - func.call
    // - any memref allocation or deallocation
    // This is conservative as there are obviously functions that work safely on the device,
    // but an inlining pass could work around that easily
    bool canBeOffloaded = true;
    op->walk([&](func::CallOp) {
        canBeOffloaded = false;
    });
    op->walk([&](memref::AllocOp) {
        canBeOffloaded = false;
    });
    op->walk([&](memref::AllocaOp) {
        canBeOffloaded = false;
    });
    op->walk([&](memref::DeallocOp) {
        canBeOffloaded = false;
    });
    op->walk([&](memref::ReallocOp) {
        canBeOffloaded = false;
    });
    kokkos::ExecutionSpace exec = canBeOffloaded ? kokkos::ExecutionSpace::Device : kokkos::ExecutionSpace::Host;
    // Possible cases for exec == Device:
    //
    // - Depth 1: RangePolicy (or MDRangePolicy, both have same representation in the dialect)
    // - Depth 2: TeamPolicy with one thread (simd) per inner work-item (best for spmv-like patterns)
    //            TODO: Write a heuristic to choose TeamPolicy/TeamVector instead, for when the inner loop
    //            requires more parallelism
    // - Depth 3: TeamPolicy/TeamThread/ThreadVector nested parallelism
    // - Depth >3: Use TeamPolicy/TeamThread for outermost two loops, and ThreadVector for innermost loop.
    //             Better coalescing that way, if data layout is correct for the loop structure.
    //             Serialize all other loops by replacing them with scf.for.
    //
    // For exec == Host, just parallelize the outermost loop with RangePolicy and serialize the inner loops.
    if(depth == 1)
    {
    }
    else if(depth == 2)
    {
    }
    else if(depth >= 3)
    {
      //TODO
    }
    if(depth > 3)
    {
      //TODO
    }
  }
};

} // namespace

void mlir::populateKokkosLoopMappingPatterns(RewritePatternSet &patterns)
{
  patterns.add<KokkosLoopRewriter>(patterns.getContext());
}

