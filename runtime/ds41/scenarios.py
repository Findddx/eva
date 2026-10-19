"""Routing probes for the three decode workloads.

Counts from these prompts decide which experts are pinned in VRAM and RAM.
"""

CODE = """\
def repair_failing_test(repo_path, test_node):
    \"\"\"DeepSWE-style fix: reproduce, patch the smallest function, add a regression test.
    The sandbox runs pytest on the named node. Read the traceback, locate the
    offending branch, and keep the public API stable. Do not rewrite the module.
    \"\"\"
    import subprocess, pathlib
    log = subprocess.run(
        ["pytest", "-q", test_node, "--tb=short"],
        cwd=repo_path, capture_output=True, text=True,
    )
    trace = log.stdout + log.stderr
    src = pathlib.Path(repo_path)
    # Prefer the frame that sits inside the repository, not site-packages.
    frames = [line for line in trace.splitlines() if str(src) in line and ".py" in line]
    target = frames[-1] if frames else ""
    return {"passed": log.returncode == 0, "trace": trace[-4000:], "target": target}

def apply_patch(path, old, new):
    file = pathlib.Path(path)
    file.write_text(file.read_text().replace(old, new, 1))
"""

PAPER = """\
混合专家推理的带宽瓶颈不在注意力的算力，而在每个 token 实际激活的专家权重能否留在显存。
对 DeepSeek-V4.1-Flash 而言，解码阶段 40 层、每层 top-6，单步要触及约三个 GiB 的低比特专家。
若这些专家已经按代码、论文和说明文三类语料的路由频率预热，则 PCIe 只承担长尾。
压缩稀疏注意力把全局 KV 压到每个 token 数百字节，因此把上下文预留到 256K 几乎不挤占专家槽。
实验应同时报告冷启动与热缓存：前者暴露 NVMe，后者才是交互解码的稳态。
我们在固定温度贪心下比较命中率、每 token 毫秒和显存占用，而不是只报一次短提示的首 token。
若把问题从写论文换成核对实验，步骤是一样的：先固定协议和对照，再看哪一个变量真的改了结果。
例如比较两组专家驻留策略时，要同时记下命中率、搬运字节和每步毫秒，不能只报一个更快的总数。
样本不够时先说明不确定性，而不是把一次短跑写成结论。
"""

PROSE = """\
把一件复杂的事讲清楚，先给结论，再给读者能核对的依据。
句子不宜叠太多的抽象名词。能用一个具体动作说完的，就不要换成“进行某种机制上的优化”。
写给同事的说明要让人知道下一步做什么：哪一份数据、哪一个命令、哪一个数字。
文学性的段落则相反，它允许停顿，允许把一个画面放慢。
这两种文字不能用同一套热度去服务。代码专家、论文专家和说明文专家本来就不是同一批人。
所以缓存要按场景预热，而不是指望一次闲聊就把全部路由学完。
"""


def texts():
    return (("code", CODE), ("science", PAPER), ("prose", PROSE))
