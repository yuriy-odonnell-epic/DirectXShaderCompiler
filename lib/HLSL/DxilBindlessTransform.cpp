// clang-format off
// DXC codebase does not have correct header include directives and 
// is sensitive to include order. Clang format must be disabled.
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilInstructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/Local.h"
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <array>
#include <vector>
// clang-format on

using namespace llvm;
using namespace hlsl;

#pragma warning(disable : 4189)

// Transforms shader with traditional "bindful" resources into a bindless form

namespace {

static inline std::string unmangle(StringRef name) {
  if (!name.startswith("\x1?"))
    return name;

  size_t pos = name.find("@@");
  if (pos == name.npos)
    return name;

  return name.substr(2, pos - 2);
}

class DxilBindlessTransform : public ModulePass {
public:
  static char ID;
  explicit DxilBindlessTransform() : ModulePass(ID) {}

  void applyOptions(PassOptions O) override {}

  const char *getPassName() const override { return "DXIL bindless transform"; }

  struct ResourceInfo {
    unsigned RangeID = UINT_MAX;
    llvm::Constant *Symbol = nullptr;
    GlobalVariable *GV = nullptr;
  };

  Type *getOrCreateType(Module &M, const char *name) {
    Type *type = M.getTypeByName(name);
    if (type == nullptr) {
      SmallVector<Type *, 1> Elements{Type::getInt32Ty(M.getContext())};
      type = StructType::create(Elements, name);
    }
    return type;
  }

  ResourceInfo addBinding(Module &M, const char *resourceName,
                          DXIL::ResourceKind kind, unsigned reg,
                          unsigned rangeSize, bool needGlobalVariable) {
    DxilModule &DM = M.GetOrCreateDxilModule();
    ResourceInfo result;

    switch (kind) {
    case DXIL::ResourceKind::Sampler: {
      Type *structTy = getOrCreateType(M, "struct.SamplerState");
      ArrayType *arrayTy =
          ArrayType::get(structTy, rangeSize == UINT_MAX ? 0 : rangeSize);
      Type *type = rangeSize == 1 ? structTy : arrayTy;
      if (needGlobalVariable) {
        result.GV = M.getGlobalVariable(resourceName);
        if (!result.GV) {
          result.GV =
              cast<GlobalVariable>(M.getOrInsertGlobal(resourceName, type));
          result.GV->setAlignment(4);
          result.GV->setConstant(true);
        }
      }
      const unsigned handle = static_cast<unsigned>(DM.GetSamplers().size());
      std::unique_ptr<DxilSampler> resource = make_unique<DxilSampler>();
      resource->SetGlobalName(unmangle(resourceName));
      resource->SetSamplerKind(DxilSampler::SamplerKind::Default);
      resource->SetKind(kind);
      resource->SetID(handle);
      resource->SetSpaceID(OutputSpace);
      resource->SetLowerBound(reg);
      resource->SetRangeSize(rangeSize);
      if (result.GV) {
        resource->SetGlobalSymbol(result.GV);
      } else {
        resource->SetGlobalSymbol(UndefValue::get(type->getPointerTo()));
      }
      result.Symbol = resource->GetGlobalSymbol();
      result.RangeID = DM.AddSampler(std::move(resource));
      break;
    }
    case DXIL::ResourceKind::RawBuffer: {
      Type *type = getOrCreateType(M, "struct.ByteAddressBuffer");
      if (needGlobalVariable) {
        result.GV = M.getGlobalVariable(resourceName);
        if (!result.GV) {
          result.GV =
              cast<GlobalVariable>(M.getOrInsertGlobal(resourceName, type));
          result.GV->setAlignment(4);
          result.GV->setConstant(true);
        }
      }
      const unsigned handle = static_cast<unsigned>(DM.GetSRVs().size());
      std::unique_ptr<DxilResource> resource = make_unique<DxilResource>();
      resource->SetRW(false);
      resource->SetGlobalName(unmangle(resourceName));
      resource->SetKind(kind);
      resource->SetID(handle);
      resource->SetSpaceID(OutputSpace);
      resource->SetLowerBound(reg);
      resource->SetRangeSize(rangeSize);
      if (result.GV) {
        resource->SetGlobalSymbol(result.GV);
      } else {
        resource->SetGlobalSymbol(UndefValue::get(type->getPointerTo()));
      }
      result.Symbol = resource->GetGlobalSymbol();
      result.RangeID = DM.AddSRV(std::move(resource));
      break;
    }
    default:
      throw std::exception("Requested resource kind is not implemented");
    }
    return result;
  }

  Value *createHandleBindless(Module &M, IRBuilder<> &Builder,
                              const char *valueName,
                              DXIL::ResourceClass ResourceClass,
                              unsigned RangeID, Value *Index,
                              bool NonUniform = false) {
    DxilModule &DM = M.GetOrCreateDxilModule();
    LLVMContext &Ctx = M.getContext();
    OP *HlslOP = DM.GetOP();
    Function *function =
        HlslOP->GetOpFunc(DXIL::OpCode::CreateHandle, Type::getVoidTy(Ctx));
    Constant *opCode =
        HlslOP->GetU32Const((unsigned)DXIL::OpCode::CreateHandle);
    Constant *resourceClass = HlslOP->GetI8Const(
        static_cast<std::underlying_type<DxilResourceBase::Class>::type>(
            ResourceClass));
    Constant *rangeId = HlslOP->GetU32Const(RangeID);
    Constant *nonUniform = HlslOP->GetI1Const(NonUniform);
    return Builder.CreateCall(
        function, {opCode, resourceClass, rangeId, Index, nonUniform},
        valueName);
  }

  Value *createHandle(Module &M, IRBuilder<> &Builder, const char *valueName,
                      DXIL::ResourceClass ResourceClass, unsigned RangeID,
                      unsigned Index) {
    DxilModule &DM = M.GetOrCreateDxilModule();
    LLVMContext &Ctx = M.getContext();
    OP *HlslOP = DM.GetOP();
    Function *function =
        HlslOP->GetOpFunc(DXIL::OpCode::CreateHandle, Type::getVoidTy(Ctx));
    Constant *opCode =
        HlslOP->GetU32Const((unsigned)DXIL::OpCode::CreateHandle);
    Constant *resourceClass = HlslOP->GetI8Const(
        static_cast<std::underlying_type<DxilResourceBase::Class>::type>(
            ResourceClass));
    Constant *rangeId = HlslOP->GetU32Const(RangeID);
    Constant *index = HlslOP->GetU32Const(Index);
    Constant *nonUniform = HlslOP->GetI1Const(0);
    return Builder.CreateCall(
        function, {opCode, resourceClass, rangeId, index, nonUniform},
        valueName);
  }

  Value *loadBufferInt(Module &M, IRBuilder<> &Builder, Twine valueName,
                       Value *BufferHandle, unsigned OffsetInBytes) {
    DxilModule &DM = M.GetOrCreateDxilModule();
    LLVMContext &Ctx = M.getContext();
    OP *HlslOP = DM.GetOP();
    Function *function =
        HlslOP->GetOpFunc(DXIL::OpCode::BufferLoad, Type::getInt32Ty(Ctx));
    Constant *opCode = HlslOP->GetU32Const((unsigned)DXIL::OpCode::BufferLoad);
    Constant *offsetInBytes = HlslOP->GetU32Const(OffsetInBytes);
    Constant *unused = UndefValue::get(llvm::Type::getInt32Ty(Ctx));
    Value *call = Builder.CreateCall(
        function, {opCode, BufferHandle, offsetInBytes, unused},
        valueName.concat("Load"));
    return Builder.CreateExtractValue(call, 0, valueName);
  }

  Value *loadRawBufferInt(Module &M, IRBuilder<> &Builder, Twine valueName,
                          Value *BufferHandle, unsigned OffsetInBytes) {
    DxilModule &DM = M.GetOrCreateDxilModule();
    LLVMContext &Ctx = M.getContext();
    OP *HlslOP = DM.GetOP();
    Function *function =
        HlslOP->GetOpFunc(DXIL::OpCode::RawBufferLoad, Type::getInt32Ty(Ctx));
    Constant *opCode =
        HlslOP->GetI32Const((unsigned)DXIL::OpCode::RawBufferLoad);
    Constant *offsetInBytes = HlslOP->GetI32Const(OffsetInBytes);
    Constant *unused = UndefValue::get(llvm::Type::getInt32Ty(Ctx));
    Constant *mask = HlslOP->GetI8Const(1);
    Constant *alignment = HlslOP->GetI32Const(4);
    Value *call = Builder.CreateCall(
        function,
        {opCode, BufferHandle, offsetInBytes, unused, mask, alignment},
        valueName.concat("Load"));
    return Builder.CreateExtractValue(call, 0, valueName);
  }

  bool runOnShader(Module &M) {
    DxilModule &DM = M.GetOrCreateDxilModule();
    OP *HlslOP = DM.GetOP();

    // Inject a bindless sampler table

    ResourceInfo bindlessSamplers = addBinding(
        M, "BindlessSamplers", DXIL::ResourceKind::Sampler, 0, UINT_MAX, false);

    ResourceInfo bindingTable = addBinding(
        M, "BindingTable", DXIL::ResourceKind::RawBuffer, 0, 1, false);

    // Find all samplers in existing code

    Function *entryPointFunction = DM.GetEntryFunction();

    for (auto &block : entryPointFunction->getBasicBlockList()) {
      IRBuilder<> OuterBuilder(block.getInstList().begin());
      Value *bindingTableHandle = nullptr;
      for (Instruction &instr : block.getInstList()) {
        DxilInst_CreateHandle createHandleInstr(&instr);
        if (!createHandleInstr) {
          continue;
        }
        DXIL::ResourceClass resourceClass =
            (DXIL::ResourceClass)createHandleInstr.get_resourceClass_val();
        if (resourceClass != DXIL::ResourceClass::Sampler) {
          continue;
        }
        if (bindingTableHandle == nullptr) {
          bindingTableHandle =
              createHandle(M, OuterBuilder, "bindingTableHandle",
                           DXIL::ResourceClass::SRV, bindingTable.RangeID, 0);
        }
        ConstantInt *bindfulIndex =
            dyn_cast<ConstantInt>(createHandleInstr.get_index());
        IRBuilder<> Builder(&instr);
        Value *bindlessIndex =
            loadBufferInt(M, Builder, "bindlessIndex", bindingTableHandle,
                          unsigned(4 * bindfulIndex->getZExtValue()));
        createHandleInstr.set_index(bindlessIndex);
        createHandleInstr.set_rangeId_val(bindlessSamplers.RangeID);
        createHandleInstr.set_nonUniformIndex_val(false);
        instr.setName("bindlessSampler");
      }
    }

    DM.m_ShaderFlags.SetEnableRawAndStructuredBuffers(true);

    DM.RemoveUnusedResources();

    // RemoveUnusedResources() does not seem to patch all the createHandle calls
    // to point to the correct range IDs, so we must do it explicitly

    std::vector<std::pair<unsigned, unsigned>> samplerRemap;
    unsigned samplerRangeID = 0;
    for (auto &it : DM.GetSamplers()) {
      DxilSampler *sampler = it.get();
      unsigned existingRangeID = sampler->GetID();
      if (existingRangeID != samplerRangeID) {
        samplerRemap.push_back(std::make_pair(existingRangeID, samplerRangeID));
        sampler->SetID(samplerRangeID);
      }
      ++samplerRangeID;
    }

    for (auto &block : entryPointFunction->getBasicBlockList()) {
      IRBuilder<> OuterBuilder(block.getInstList().begin());
      Value *bindingTableHandle = nullptr;
      for (Instruction &instr : block.getInstList()) {
        DxilInst_CreateHandle createHandleInstr(&instr);
        if (!createHandleInstr) {
          continue;
        }
        if (createHandleInstr.get_resourceClass_val() !=
            (int8_t)DXIL::ResourceClass::Sampler) {
          continue;
        }
        for (const auto &it : samplerRemap) {
          if (it.first == createHandleInstr.get_rangeId_val()) {
            createHandleInstr.set_rangeId_val(it.second);
            break;
          }
        }
      }
    }

    DM.ReEmitDxilResources();

    return true;
  }

  // Adapted from DxilPatchShaderRecordBindings::GetHandleInfo()
  DxilResourceBase *extractHandleResource(Module &M,
                                          DxilInst_CreateHandleForLib &instr) {
    DxilModule &DM = M.GetOrCreateDxilModule();
    LoadInst *loadRangeId = cast<LoadInst>(instr.get_Resource());
    Value *ResourceSymbol = loadRangeId->getPointerOperand();
    DXIL::ResourceClass resourceClasses[] = {
        DXIL::ResourceClass::CBuffer, DXIL::ResourceClass::SRV,
        DXIL::ResourceClass::UAV, DXIL::ResourceClass::Sampler};

    hlsl::DxilResourceBase *resource = nullptr;
    for (auto &resourceClass : resourceClasses) {

      switch (resourceClass) {
      case DXIL::ResourceClass::CBuffer: {
        auto &cbuffers = DM.GetCBuffers();
        for (auto &cbuffer : cbuffers) {
          if (cbuffer->GetGlobalSymbol() == ResourceSymbol) {
            resource = cbuffer.get();
            break;
          }
        }
        break;
      }
      case DXIL::ResourceClass::SRV:
      case DXIL::ResourceClass::UAV: {
        auto &viewList = resourceClass == DXIL::ResourceClass::SRV
                             ? DM.GetSRVs()
                             : DM.GetUAVs();
        for (auto &view : viewList) {
          if (view->GetGlobalSymbol() == ResourceSymbol) {
            resource = view.get();
            break;
          }
        }
        break;
      }
      case DXIL::ResourceClass::Sampler: {
        auto &samplers = DM.GetSamplers();
        for (auto &sampler : samplers) {
          if (sampler->GetGlobalSymbol() == ResourceSymbol) {
            resource = sampler.get();
            break;
          }
        }
        break;
      }
      }
    }
    return resource;
  }

  void runOnLibraryFunction(Module &M, Function *Fun) {
    DxilModule &DM = M.GetOrCreateDxilModule();
    OP *HlslOP = DM.GetOP();

    const char *bindingTableName = "\01?BindingTable@@3UByteAddressBuffer@@A";
    // const char *bindingTableName = "BindingTable";
    ResourceInfo bindingTable = addBinding(
        M, bindingTableName, DXIL::ResourceKind::RawBuffer, 0, 1, true);

    const char *bindlessSamplersName =
        "\01?BindlessSamplers@@3PAUSamplerState@@A";
    // const char *bindlessSamplersName = "BindlessSamplers";
    ResourceInfo bindlessSamplers =
        addBinding(M, bindlessSamplersName, DXIL::ResourceKind::Sampler, 0,
                   UINT_MAX, true);

    for (auto &block : Fun->getBasicBlockList()) {
      Value *bindingTableHandle = nullptr;
      std::vector<llvm::Instruction *> toRemove;
      // TODO: use BasicBlock::getFirstNonPhi() or
      // BasicBlock::getFirstNonPHIOrDbgOrLifetime() to ensure that // the
      // builder is created at a valid point.
      IRBuilder<> OuterBuilder(block.getInstList().begin());

      // TODO: instead of iterating over all instructions, it should be possible
      // to use approach similar to DxilCondenseResources::PatchCreateHandle
      // Tex Riddell:
      // This is more complicated for createHandleForLib due to overloading.  I
      // would iterate functions, skipping ones that are llvm intrinsics, aren't
      // declarations only, are unused, and not dxil intrinsic functions.  Then
      // iterate users and try making a DxilInst_CreateHandleForLib with it to
      // see if it's the right thing, if not, break.  Then you can process each
      // case. Create builder to insert before each createHandle call location
      // being modified and insert the load there.
      // Create one handle for your new rawbuffer per function in entry block,
      // keeping track with a map of function -> createHandle.

      for (Instruction &instr : block.getInstList()) {
        DxilInst_CreateHandleForLib createHandleForLibInstr(&instr);
        if (!createHandleForLibInstr) {
          continue;
        }
        DxilResourceBase *resource =
            extractHandleResource(M, createHandleForLibInstr);
        if (!resource) {
          continue;
        }
        if (resource->GetSpaceID() != InputSpace) {
          continue;
        }
        if (resource->GetClass() != DXIL::ResourceClass::Sampler) {
          continue;
        }

        IRBuilder<> Builder(&instr);

        if (bindingTableHandle == nullptr) {
          LoadInst *loadInst =
              OuterBuilder.CreateLoad(bindingTable.GV, "bindingTableLoad");
          loadInst->setAlignment(4);
          Function *function = HlslOP->GetOpFunc(
              DXIL::OpCode::CreateHandleForLib, loadInst->getType());
          Constant *opCode =
              HlslOP->GetU32Const((unsigned)DXIL::OpCode::CreateHandleForLib);
          bindingTableHandle = Builder.CreateCall(function, {opCode, loadInst},
                                                  "bindingTableHandle");
        }

        unsigned bindfulIndex = resource->GetLowerBound();
        Value *bindlessIndex =
            loadRawBufferInt(M, Builder, "bindlessIndex", bindingTableHandle,
                             unsigned(4 * bindfulIndex));
        Value *bindlessSamplerGEP = Builder.CreateInBoundsGEP(
            bindlessSamplers.GV, {HlslOP->GetU32Const(0), bindlessIndex},
            "bindlessSamplerGEP");
        LoadInst *bindlessSampler = Builder.CreateLoad(bindlessSamplerGEP);
        bindlessSampler->setAlignment(4);

        Value *oldResource = createHandleForLibInstr.get_Resource();
        if (auto oldResourceInst = cast<Instruction>(oldResource)) {
          toRemove.push_back(oldResourceInst);
        }

        CallInst *callInst = cast<CallInst>(createHandleForLibInstr.Instr);
        callInst->setCalledFunction(HlslOP->GetOpFunc(
            DXIL::OpCode::CreateHandleForLib, bindlessSampler->getType()));
        createHandleForLibInstr.set_Resource(bindlessSampler);
      }
      for (auto it : toRemove) {
        it->eraseFromParent();
      }
    }
  }

  bool runOnLibrary(Module &M) {
    for (auto F = M.begin(); F != M.end(); ++F) {
      if (F->isDeclaration()) {
        continue;
      }
      runOnLibraryFunction(M, F);
    }
    DxilModule &DM = M.GetOrCreateDxilModule();

    DM.m_ShaderFlags.SetEnableRawAndStructuredBuffers(true);
    DM.RemoveResourcesWithUnusedSymbols();

    DM.ReEmitDxilResources();
    return true;
  }

  bool runOnModule(Module &M) override {
    DxilModule &DM = M.GetOrCreateDxilModule();
    const ShaderModel *SM = DM.GetShaderModel();
    if (SM->IsLib()) {
      return runOnLibrary(M);
    } else {
      return runOnShader(M);
    }
  }

private:
  unsigned InputSpace = 0;    // TODO: make configurable
  unsigned OutputSpace = 100; // TODO: make configurable
};

char DxilBindlessTransform::ID = 0;

} // namespace

ModulePass *llvm::createDxilBindlessTransformPass() {
  return new DxilBindlessTransform();
}

INITIALIZE_PASS(DxilBindlessTransform, "dxil-bindless-transform",
                "DXIL bindless transform", false, false)
