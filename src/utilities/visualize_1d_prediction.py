# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pandas as pd
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Select
from bokeh.plotting import figure


def visualize(filename):
    df = pd.read_csv(filename)
    ds = ColumnDataSource(df)
    p = figure(title="title", toolbar_location="above", x_axis_type="linear")

    line_renderer = p.line('x', 'ground_truth', source=ds)
    handler = CustomJS(args=dict(line_renderer=line_renderer),
                       code="""
        line_renderer.glyph.y = {field: cb_obj.value};
    """)

    select = Select(title="Model Type:", options=list(df.columns))
    select.js_on_change('value', handler)
    select.js_link('value', p.title, 'text')
    show(column(select, p))


if __name__ == "__main__":
    filename = sys.argv[1]
    visualize(filename)
