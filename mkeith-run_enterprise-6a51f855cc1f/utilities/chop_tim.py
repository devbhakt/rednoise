#!/usr/bin/env python


import sys

par=sys.argv[1]
tim=sys.argv[2]
chop_range=5000

parlines=[]
glitches={}
glepoch=[] # mod ly
with open(par) as f:
    for line in f:
        if line.startswith("GL"):
            e=line.split()
            pp = e[0].split("_")
            i = int(pp[1])
            param=pp[0]
            if not i in glitches:
                glitches[i] = {'turns':0}
            if param=="GLEP":
                glitches[i]['epoch'] = float(e[1])
            if param=="GLPH":
                glitches[i]['turns'] = round(float(e[1]))
                e[1] = "{}".format(float(e[1]) - glitches[i]['turns'])
                continue

            glitches[i][param] = " ".join(e[1:])

        else:
            parlines.append(line)


for ig in glitches:
    print("glitch[{}] epoch {} turns {}".format(ig,glitches[ig]['epoch'],glitches[ig]['turns']))
    glepoch.append(glitches[ig]['epoch']) # mod ly


gg = sorted(glitches,key=lambda x: glitches[x]['epoch'])
print("Time-order is",gg)

glepoch.sort()
print("All GLEP:", glepoch)

i=1
for ig in gg:
    glitches[ig]['newid'] = i
    i+=1

timlines=[]
with open(tim) as f:
    for line in f:
        e=line.split()
        if "-pn" in e:
            epoch = float(e[2])
            ii = e.index("-pn")
            pn = int(e[ii+1])
            for ig in gg:
                if epoch > glitches[ig]['epoch']:
                    pn -= glitches[ig]['turns']
            newline = " ".join(e[:ii])+" -pn {} ".format(pn)+(" ".join(e[ii+2:]))
            if glepoch[0]-chop_range <= epoch <= glepoch[-1]+chop_range:
                timlines.append(" "+newline+"\n")
        else:
            if len(e)>3:
                epoch = float(e[3])
                if glepoch[0]-chop_range <= epoch <= glepoch[-1]+chop_range:
                    timlines.append(line)
            else:
                timlines.append(line)


with open("chp_"+tim,"w") as f:
    f.writelines(timlines)

for ig in gg:
    if "GLF0" not in glitches[ig]:
        nowid = glitches[ig]['newid']
        print("Glitch[{}] is a double recovery".format(ig))
        idxpre = next((idx for idx in glitches if glitches[idx]['newid'] == nowid-1), None)
        idxpost = next((idx for idx in glitches if glitches[idx]['newid'] == nowid+1), None)
        if glitches[ig]['epoch'] == glitches[idxpre]['epoch']:
            print("Merge glitch[{}] to double recovery glitch[{}]".format(ig, idxpre))
            if float(glitches[ig]['GLTD']) < float(glitches[idxpre]['GLTD']):
                print("Sort recovery of glitch[{}] as the first recovery of glitch[{}]".format(ig, idxpre))
                glitches[idxpre]['GLF0D2'] = glitches[idxpre]['GLF0D']
                glitches[idxpre]['GLTD2'] = glitches[idxpre]['GLTD']
                glitches[idxpre]['GLF0D'] = glitches[ig]['GLF0D']
                glitches[idxpre]['GLTD'] = glitches[ig]['GLTD']
            else:
                glitches[idxpre]['GLF0D2'] = glitches[ig]['GLF0D']
                glitches[idxpre]['GLTD2'] = glitches[ig]['GLTD']
            del glitches[ig]
        elif glitches[ig]['epoch'] == glitches[idxpost]['epoch']:
            print("Merge glitch[{}] to double recovery glitch[{}]".format(ig, idxpost))
            if float(glitches[ig]['GLTD']) < float(glitches[idxpost]['GLTD']):
                print("Sort recovery of glitch[{}] as the first recovery of glitch[{}]".format(ig, idxpost))
                glitches[idxpost]['GLF0D2'] = glitches[idxpost]['GLF0D']
                glitches[idxpost]['GLTD2'] = glitches[idxpost]['GLTD']
                glitches[idxpost]['GLF0D'] = glitches[ig]['GLF0D']
                glitches[idxpost]['GLTD'] = glitches[ig]['GLTD']
            else:
                glitches[idxpost]['GLF0D2'] = glitches[ig]['GLF0D']
                glitches[idxpost]['GLTD2'] = glitches[ig]['GLTD']
            del glitches[ig]
        else:
            print("Glitch[{}] cannot merge as double recovery of glitch[{}] or [{}]".format(ig, idxpre, idxpost))

newgg = sorted(glitches,key=lambda x: glitches[x]['epoch'])
print("New time-order is",newgg)

j=1
for ig in newgg:
    glitches[ig]['newid'] = j
    j+=1


with open("dbr_"+par,"w") as f:
    f.writelines(parlines)
    for ig in newgg:
        for param in glitches[ig]:
            if param in ["epoch","turns","newid"]:
                continue
            f.write("{}_{} {}\n".format(param,glitches[ig]['newid'],glitches[ig][param]))



