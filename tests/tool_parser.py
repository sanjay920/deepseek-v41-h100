"""Check the installed V4.1 parser against native calls and stream boundaries."""
import json
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

TOOLS=[Tool.model_validate({'type':'function','function':{'name':name,'parameters':{'type':'object'}}})
       for name in ('bash','glob','emit_payload','get_weather','status')]
cases=[
    ('harness_parallel','<｜DSML｜ calls> <｜DSML｜ invoke name="bash"> <｜DSML｜ parameter name="command" string="true">pwd && ls -la</｜DSML｜ parameter> <｜DSML｜ parameter name="description" string="true">List workspace root contents</｜DSML｜ parameter> </｜DSML｜ invoke> <｜DSML｜ invoke name="glob"> <｜DSML｜ parameter name="pattern" string="true">*</｜DSML｜ parameter> </｜DSML｜ invoke> </｜DSML｜ calls>',
     [('bash',{'command':'pwd && ls -la','description':'List workspace root contents'}),('glob',{'pattern':'*'})]),
    ('typed','<｜DSML｜ calls>\n<｜DSML｜ invoke name="emit_payload">\n<｜DSML｜ parameter name="count" string="false">3</｜DSML｜ parameter>\n<｜DSML｜ parameter name="enabled" string="false">true</｜DSML｜ parameter>\n<｜DSML｜ parameter name="items" string="false">["é", "日本語"]</｜DSML｜ parameter>\n<｜DSML｜ parameter name="data" string="false">{"value":null}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>',
     [('emit_payload',{'count':3,'enabled':True,'items':['é','日本語'],'data':{'value':None}})]),
    ('json_body','<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">{"city":"Paris"}</｜DSML｜ invoke>\n</｜DSML｜ calls>',[('get_weather',{'city':'Paris'})]),
    ('zero_args','<｜DSML｜ calls>\n<｜DSML｜ invoke name="status"/>\n</｜DSML｜ calls>',[('status',{})]),
]


def stream(chunks):
    parser=FunctionCallParser(TOOLS,'deepseekv41')
    normal='';calls={}
    pieces=[parser.parse_stream_chunk(c) for c in chunks]+[parser.parse_stream_end()]
    for text,items in pieces:
        normal+=text
        for item in items:
            slot=calls.setdefault(item.tool_index,{'name':None,'arguments':''})
            if item.name:slot['name']=item.name
            slot['arguments']+=item.parameters
    return normal,[(c['name'],json.loads(c['arguments'])) for _,c in sorted(calls.items())]


results=[]
for name,text,expected in cases:
    normal,calls=FunctionCallParser(TOOLS,'deepseekv41').parse_non_stream(text)
    assert [(c.name,json.loads(c.parameters)) for c in calls]==expected,(name,calls)
    assert not normal.strip(),(name,normal)
    variants=[[text],list(text)]+[[text[:i],text[i:]] for i in range(len(text)+1)]
    for chunks in variants:
        normal,parsed=stream(chunks)
        assert parsed==expected,(name,chunks,parsed)
        assert not normal.strip(),(name,normal)
    results.append({'name':name,'passed':True,'stream_chunkings':len(variants)})
out={'passed':True,'parser':'deepseekv41','results':results}
print(json.dumps(out,indent=2))
