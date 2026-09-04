# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract the system prompt, first-user wrapper (seeded MEMORY.md system-reminder) and tool schemas from a proxy wire dump.

Usage: python wire_template.py <proxy_wire.jsonl> <template.json>; ``<RUN>`` marks the run/task dir.
"""
import json,sys,re,os
f=sys.argv[1]; out=sys.argv[2]
for l in open(f):
    r=json.loads(l)
    if r["kind"]=="request" and r["body"].get("tools"):
        b=r["body"]; break
ms=b["messages"]; sysp=ms[0]["content"]; u=ms[1]["content"]
utext="\n".join(x["text"] for x in u if x.get("type")=="text") if isinstance(u,list) else u
run=re.search(r"/mnt/data/zhihan/reviewer_reduce100/runs/[^/ ]+/task-\d{3}",sysp).group(0)
print("run path:",run); print("occurrences in system:",sysp.count(run),"in user1:",utext.count(run))
print("other abs paths in system:",sorted(set(re.findall(r"/(?:mnt|home)/[^\s'\")\]]+",sysp.replace(run,"<RUN>")))))
print("other abs paths in user1:",sorted(set(re.findall(r"/(?:mnt|home)/[^\s'\")\]]+",utext.replace(run,"<RUN>"))))[:10])
i=utext.find("TASK:"); print("user1: system-reminder part len",i,"task part len",len(utext)-i)
print("user1 head:",utext[:300].replace("\n","|")); print("user1 around TASK:",utext[i-300:i+80].replace("\n","|"))
json.dump({"system":sysp.replace(run,"<RUN>"),"user1_prefix":utext[:i].replace(run,"<RUN>"),"tools":b["tools"],"model":b.get("model")},open(out,"w"))
print("dates in system:",re.findall(r"20\d\d-\d\d-\d\d",sysp)[:5]); print("saved",out)
