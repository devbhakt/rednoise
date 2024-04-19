#!/usr/bin/env python

import sys,os
from uncertainties import ufloat

LATEX_PRE=r"""
\documentclass{article}
\usepackage[a4paper,margin=1cm]{geometry}
\usepackage{graphicx}
\usepackage{hyperref}

\usepackage{booktabs,array}
\newcount\rowc

\makeatletter
\def\ttabular{%
\hbox\bgroup
\let\\\cr
\def\rulea{\ifnum\rowc=\@ne \hrule height 0pt \fi}
\def\ruleb{
\ifnum\rowc=1\hrule height 1pt \else
\ifnum\rowc=3\hrule height 0.5pt \else
\ifnum\rowc=4\hrule height 0.5pt \else
\ifnum\rowc=5\hrule height 0.5pt \else
\hrule height 0pt \fi\fi\fi\fi}
\valign\bgroup
\global\rowc\@ne
\rulea
\hbox to 10em{\strut \hfill##\hfill}%
\ruleb
&&%
\global\advance\rowc\@ne
\hbox to 10em{\strut\hfill##\hfill}%
\ruleb
\cr}
\def\endttabular{%
\crcr\egroup\egroup}

\begin{document}

"""
LATEX_POST=r"""
\end{document}

"""


def run(psr, rootdir = "/nvme1/yliu/yangliu/yang/models/"):

    prefixes = "bst"
    suffixes = {0:"f", 1:"r", 2:"d", 3:"t"}
    postfixes = {"std":"n", "glf2":"y"}

    table_pars = ["GLEP", "GLF0", "GLF1", "GLF2", "GLF0D", "GLTD", "GLF0D2", "GLTD2", "GLF0D3", "GLTD3"]

    with open("{}_sum.tex".format(psr),"w") as outfile:
        outfile.write(LATEX_PRE)
        outfile.write("\\section{{{}}}".format(psr))

        figures = {}
        figure_notes = {}

        table_started = False
        for glitch in range(100):
            glitch_started = False
            proot = os.path.join(rootdir, psr)#
            for recoveries in [0, 1, 2, 3]:
                for ty in ["std", "glf2"]:
                    if glitch == 0:
                        glid = ""
                        glitch_val = 1
                    else:
                        glid = "{}".format(glitch)
                        glitch_val = glitch
                    suffix = "{}{}{}".format(glid, suffixes[recoveries], postfixes[ty])
                    res_file = os.path.join(proot,"{}_{}_{}.par.results".format(prefixes, psr, suffix))
                    if os.path.exists(res_file):
                        print("Found results for {} #{} {} {}".format(psr, glitch, ty, recoveries))
                        results = read_results_file(res_file)
                    else:
                        retults = None
                        continue

                    corner_pdf = os.path.join(proot,"{}_{}.corner.pdf".format(psr, suffix))
                    if os.path.exists(corner_pdf):
                        fig_ref = "{}_{}_{}".format(prefixes, psr, suffix)
                        figures[fig_ref] = corner_pdf
                        figure_notes[fig_ref] = "{} Glitch {}, Model: {} {}".format(psr, glitch_val, ty, recoveries)
                    else:
                        fig_ref = None
                        print(" >> missing {}".format(corner_pdf))

                    ev_file = os.path.join(proot, "{}_{}".format(psr,suffix),"pmn-stats.dat")
                    if os.path.exists(ev_file):
                        with open(ev_file) as f:
                            line = f.readline() # Discard Nested Sampling Global Log-Evidence
                            line = f.readline() # Adopt Nested Importance Sampling Global Log-Evidence
                            lnev = ufloat(float(line.split()[5]), float(line.split()[7]))
                            if ty == "std" and recoveries == 0:
                                firstev = lnev
                            difflnev = lnev - firstev
                    else:
                        lnev = None
                        difflnev = None
                        print(" >> missing {}".format(ev_file))

                    if not glitch_started:
                        if table_started:
                            outfile.write(r"\end{ttabular}\end{table}")
                            outfile.write("\n")
                            table_started = False
                        outfile.write("\subsection{{Glitch {}}}\n".format(glitch_val))
                        glitch_started = True

                    if not table_started:
                        tabformat = "ccc|cc|"+"".join(["c"]*len(table_pars))
                        outfile.write(r"\begin{table}[h]\tiny\begin{ttabular}")
                        #outfile.write("{{{}}}".format(tabformat))
                        outfile.write("\n")
                        outfile.write(r" & &  $N_\mathrm{exp}$   &     lnZ & $lnZ-lnZ_{0}$ &")
                        outfile.write(" & ".join(table_pars))
                        outfile.write(" & \\\\\n ") # End of line
                        table_started = True
                    if fig_ref is None:
                        outfile.write("-- &")
                    else:
                        outfile.write("\\ref{{{}}} &".format(fig_ref))
                    if ty == "glf2":
                        outfile.write("Y&")
                    else:
                        outfile.write("N&")
                    outfile.write("{} & ".format(recoveries))
                    if lnev is None:
                        outfile.write("-- &")
                    else:
                        outfile.write("${:.1uSL}$ &".format(lnev))
                    if difflnev is None:
                        outfile.write("-- &")
                    else:
                        outfile.write("${:.1uSL}$ &".format(difflnev))

                    cc = []
                    for col in table_pars:
                        param = "{}_{}".format(col,glitch_val)
                        if param in results:
                            cc.append("${:.1uSL}$".format(results[param]))
                        else:
                            cc.append("--")
                    outfile.write(" & ".join(cc))
                    outfile.write(" & \\\\\n") # End of line

        if table_started:
            outfile.write(r"\end{ttabular}\end{table}")
            outfile.write("\n")
            table_started = False
           
        comments = os.path.join(rootdir, psr, "{}_comments.txt".format(psr))#
        if os.path.exists(comments):
            with open(comments) as f:
                outfile.writelines(f.readlines())
        else:
            print("\n\n >> No comments file {}".format(comments))

        nunudot = os.path.join(rootdir, psr, "full_nu_nudot_gp_{}.pdf".format(psr))#
        if os.path.exists(nunudot):
            outfile.write(r"\begin{figure}[b]")
            outfile.write("\n")
            outfile.write("\includegraphics[width=\\textwidth,height=0.95\\textheight,keepaspectratio]{{{}}}\n".format(nunudot))
            outfile.write(r"\end{figure}")
            outfile.write("\n")

        for fig in figures:
            outfile.write(r"\clearpage{}")
            outfile.write(r"\begin{figure}")
            outfile.write("\n")
            e = figures[fig].rsplit(".",maxsplit=1)
            outfile.write("\includegraphics[width=\\textwidth,height=0.95\\textheight,keepaspectratio]{{{{{}}}.{}}}\n".format(e[0],e[1]))
            outfile.write("\caption{{\label{{{}}} {} }}".format(fig,figure_notes[fig]))
            outfile.write(r"\end{figure}")
            outfile.write("\n")

        outfile.write(LATEX_POST)


def read_results_file(res_file):
    results={}
    with open(res_file) as f:
        f.readline()
        for line in f:
            e=line.strip().rsplit(maxsplit=8)
            results[e[0]] = ufloat(float(e[2]),float(e[3]))
    return results


if __name__=="__main__":
    psr=sys.argv[1]
    run(psr)
