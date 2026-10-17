"""Compile project/tokens.json to tokens.css (the layout the Design System page uses).

Used for local previews and as the research UI's stylesheet source."""
import json, re, sys
from pathlib import Path

def compile_tokens(t: dict) -> str:
    out = [f"/* {t['name']} — generated from tokens.json */"]
    first = t["color"]["themes"][0]["id"]
    def cval(v):
        v = v if isinstance(v, str) else v[first]
        m = re.fullmatch(r"\{(.+)\}", v)
        return f"var(--{m.group(1)})" if m else v
    out.append(f':root, [data-theme="{first}"] {{')
    out += [f"  --{c['name']}: {cval(c['value'])};" for c in t["color"]["tokens"]]
    out.append("}")
    out.append(":root {")
    for fam in ("spacing", "radius"):
        out += [f"  --{x['name']}: {x['value']};" for x in t[fam]["tokens"]]
    out += [f"  --font-{k}: {v};" for k, v in t["type"]["families"].items()]
    out.append("}")
    for g in t["type"]["groups"]:
        for s in g["styles"]:
            fam = s.get("family", g["family"])
            decl = [f"font-family: var(--font-{fam})", f"font-size: {s['fontSize']}",
                    f"line-height: {s['lineHeight']}", f"font-weight: {s['fontWeight']}"]
            if "letterSpacing" in s:
                decl.append(f"letter-spacing: {s['letterSpacing']}")
            out.append(f".{s['name']} {{ {'; '.join(decl)}; }}")
    return "\n".join(out) + "\n"

if __name__ == "__main__":
    src = Path(sys.argv[1])
    print(compile_tokens(json.loads(src.read_text())), end="")
