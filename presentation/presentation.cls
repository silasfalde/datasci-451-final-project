\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{presentation}[2024/12/25 Presentation class]
\LoadClass{beamer}

\RequirePackage{
	amsthm,
	amsmath,
	amssymb,
	xcolor,
	geometry,
	graphicx,
	csvsimple,
	tikz,
	inputenc,
}

\input{math-commands}
\input{programming-commands}

\author{Silas Falde}
\institute{University of Michigan}
\logo{\raisebox{-0.3cm}{\includegraphics[height=0.25cm]{umich-logo.png}}}

\usetheme{Berlin}
\usecolortheme{wolverine}
\usetikzlibrary{shapes.symbols}

\AtBeginSection[]
{
	\begin{frame}
		\frametitle{Table of Contents}
		\tableofcontents[currentsection]
	\end{frame}
}

\setbeamertemplate{navigation symbols}{}

