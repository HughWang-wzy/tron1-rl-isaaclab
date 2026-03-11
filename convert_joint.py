from pxr import UsdPhysics, Sdf, Gf
import omni
stage = omni.usd.get_context().get_stage()

old_joint_path = Sdf.Path("/World/K2/left_longlink_joint")
new_joint_path = Sdf.Path("/World/K2/left_longlink_joint")

# -------------------------
# 1. 读取旧关节
# -------------------------
old_prim = stage.GetPrimAtPath(old_joint_path)
if not old_prim.IsValid():
    raise RuntimeError("旧关节不存在")

old_joint = UsdPhysics.Joint(old_prim)

# 读取位置（如果不存在则给默认值）
localPos0 = (
    old_joint.GetLocalPos0Attr().Get()
    if old_joint.GetLocalPos0Attr().HasAuthoredValue()
    else Gf.Vec3f(0, 0, 0)
)

localPos1 = (
    old_joint.GetLocalPos1Attr().Get()
    if old_joint.GetLocalPos1Attr().HasAuthoredValue()
    else Gf.Vec3f(0, 0, 0)
)

# （可选）读取旋转
localRot0 = (
    old_joint.GetLocalRot0Attr().Get()
    if old_joint.GetLocalRot0Attr().HasAuthoredValue()
    else Gf.Quatf(1, 0, 0, 0)
)

localRot1 = (
    old_joint.GetLocalRot1Attr().Get()
    if old_joint.GetLocalRot1Attr().HasAuthoredValue()
    else Gf.Quatf(1, 0, 0, 0)
)

# 读取绑定刚体（防止手写错）
body0 = old_joint.GetBody0Rel().GetTargets()[0]
body1 = old_joint.GetBody1Rel().GetTargets()[0]

# -------------------------
# 2. 删除旧关节
# -------------------------
stage.RemovePrim(old_joint_path)

# -------------------------
# 3. 创建球关节
# -------------------------
joint = UsdPhysics.SphericalJoint.Define(stage, new_joint_path)

# 绑定刚体
joint.CreateBody0Rel().SetTargets([body0])
joint.CreateBody1Rel().SetTargets([body1])

# 设置旧关节位置
joint.CreateLocalPos0Attr().Set(localPos0)
joint.CreateLocalPos1Attr().Set(localPos1)

# 设置旧关节旋转（强烈建议）
joint.CreateLocalRot0Attr().Set(localRot0)
joint.CreateLocalRot1Attr().Set(localRot1)

print("RevoluteJoint → SphericalJoint 转换完成")
