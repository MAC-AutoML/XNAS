from collections import namedtuple
from graphviz import Digraph
import sys

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 3), ('dil_conv_5x5', 2), ('avg_pool_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 3), ('dil_conv_5x5', 2), ('avg_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5])

def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=False)


if __name__ == '__main__':
#   try:
#     genotype = eval('genotypes.{}'.format(genotype))
#   except AttributeError:
#     print("{} is not specified in genotypes.py".format(genotype_name)) 
#     sys.exit(1)

  plot(genotype.normal, "normal")
  plot(genotype.reduce, "reduction")