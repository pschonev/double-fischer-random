# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "d2-widget==0.0.3",
#     "graphviz==0.21",
#     "graphviz-anywidget==0.8.0",
#     "plotly==6.5.0",
#     "pyecharts==2.0.9",
#     "python-igraph==1.0.0",
#     "pyvis==0.3.2",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from graphviz_anywidget import graphviz_widget_simple
    return


@app.cell
def _():
    import msgspec

    # -----------------------------------------------------------------------------
    # 1. Data Structures
    # -----------------------------------------------------------------------------


    class PositionAnalysis(msgspec.Struct):
        cpl: int | None = None
        mate: int | None = None
        pv: list[str] | None = None


    class PositionNode(msgspec.Struct):
        """Node in the position analysis tree which represents a halfmove (ply).
        'move' is the move that led to this position (e.g. 'e4').
        """

        move: str
        children: list["PositionNode"]
        analysis: PositionAnalysis


    # -----------------------------------------------------------------------------
    # 2. Visualization Logic
    # -----------------------------------------------------------------------------


    def node_to_dot_source(root: PositionNode, root_is_white: bool = True) -> str:
        """
        Traverses the PositionNode tree and returns a DOT string suitable for Graphviz.
        """
        lines = [
            "digraph {",
            "  rankdir=TB;",
            '  node [fontname="Courier", shape="box", style="filled"];',
            '  edge [fontname="Helvetica"];',
        ]

        # Helper to format the evaluation string
        def get_eval_label(analysis: PositionAnalysis) -> str:
            if analysis.mate is not None:
                return f"M{analysis.mate}"
            if analysis.cpl is not None:
                # Convert centipawns to pawn units for display
                return f"{analysis.cpl / 100:.2f}"
            return "?"

        # Use a unique ID counter for graph nodes
        node_counter = 0

        # Stack for DFS: (current_node, parent_id, is_white_turn)
        # We assume the root represents the position BEFORE the move,
        # or the result of a null move if it's the start.
        # But based on your struct, 'move' is the edge leading here.

        root_id = f"node_{node_counter}"
        node_counter += 1

        # Color logic: gold for White to move, burlywood4 for Black to move
        color = "gold" if root_is_white else "burlywood4"
        label = get_eval_label(root.analysis)

        lines.append(f'  {root_id} [label="{label}", fillcolor="{color}"];')

        # Stack: (node_obj, graphviz_id, is_white_turn)
        stack = [(root, root_id, root_is_white)]

        while stack:
            curr, parent_id, white_to_move = stack.pop()

            # Sort children by best eval to keep graph tidy (optional)
            # Assuming standard engine output where evaluation is relative to side to move?
            # Or absolute? Let's just process them in order.

            for child in curr.children:
                child_id = f"node_{node_counter}"
                node_counter += 1

                # Flip turn for the next node
                child_is_white = not white_to_move

                # Node styling
                fill = "gold" if child_is_white else "burlywood4"
                lbl = get_eval_label(child.analysis)

                # Determine if this is a PV (Principal Variation) node roughly
                # You could add logic here to bold the best move

                lines.append(f'  {child_id} [label="{lbl}", fillcolor="{fill}"];')

                # Edge styling
                edge_color = (
                    "gold" if white_to_move else "burlywood4"
                )  # Color of the side that moved
                lines.append(
                    f"  {parent_id} -> {child_id} "
                    f'[label="{child.move}", color="{edge_color}", fontcolor="black"];'
                )

                stack.append((child, child_id, child_is_white))

        lines.append("}")
        return "\n".join(lines)


    # -----------------------------------------------------------------------------
    # 3. Sample Tree Construction
    # -----------------------------------------------------------------------------


    def create_sample_tree() -> PositionNode:
        """Creates a mock analysis tree for the Opening position."""

        # Depth 2 (Grandchildren)
        # Variation 1: 1. e4 e5
        node_e4_e5 = PositionNode(
            move="e5",
            children=[],
            analysis=PositionAnalysis(cpl=20, pv=["Nf3", "Nc6"]),
        )
        # Variation 2: 1. e4 c5
        node_e4_c5 = PositionNode(
            move="c5",
            children=[],
            analysis=PositionAnalysis(cpl=35, pv=["Nf3", "d6"]),
        )

        # Depth 1 (Children)
        # Variation A: 1. e4
        node_e4 = PositionNode(
            move="e4",
            children=[node_e4_e5, node_e4_c5],
            analysis=PositionAnalysis(cpl=30, pv=["e5", "Nf3"]),
        )

        # Variation B: 1. d4
        node_d4 = PositionNode(
            move="d4",
            children=[
                PositionNode(
                    move="Nf6", children=[], analysis=PositionAnalysis(cpl=25)
                )
            ],
            analysis=PositionAnalysis(cpl=28, pv=["Nf6", "c4"]),
        )

        # Root Node (Starting Position)
        root = PositionNode(
            move="startpos",  # Or empty string
            children=[node_e4, node_d4],
            analysis=PositionAnalysis(cpl=35, pv=["e4", "e5"]),
        )

        return root


    # -----------------------------------------------------------------------------
    # 4. Usage in Marimo
    # -----------------------------------------------------------------------------

    # 1. Create the data
    tree = create_sample_tree()

    # 2. Generate the DOT string
    dot_source = node_to_dot_source(tree, root_is_white=True)

    # 3. Display (Uncomment the import and widget call in your notebook)
    return PositionNode, node_to_dot_source, tree


@app.cell
def _(PositionNode):
    import marimo as mo


    def node_to_mermaid(root: PositionNode, root_is_white: bool = True) -> str:
        # Start the flowchart definition
        # TD = Top Down orientation
        lines = ["graph TD"]

        # Use a counter to generate unique IDs like id0, id1...
        node_counter = 0

        # Function to format label
        def get_label(analysis):
            if analysis.mate is not None:
                return f"M{analysis.mate}"
            if analysis.cpl is not None:
                return f"{analysis.cpl / 100:.2f}"
            return "?"

        # Format: stack contains (node_obj, node_id, is_white_turn)
        root_id = f"id{node_counter}"
        node_counter += 1

        # Define the root style
        # Mermaid styling is defined at the end usually, or inline
        # We will use inline styling syntax: id["Label"]:::ClassName

        stack = [(root, root_id, root_is_white)]

        # We need to collect styles to apply at the end
        styles = []

        while stack:
            curr, pid, white_to_move = stack.pop()

            # Determine class based on turn (for styling)
            style_class = "whiteTurn" if white_to_move else "blackTurn"
            lbl = get_label(curr.analysis)

            # Add node definition if it's the root (otherwise handled in edge)
            if pid == root_id:
                lines.append(f'    {pid}("{lbl}"):::{style_class}')

            for child in curr.children:
                cid = f"id{node_counter}"
                node_counter += 1
                child_is_white = not white_to_move
                child_class = "whiteTurn" if child_is_white else "blackTurn"
                child_lbl = get_label(child.analysis)

                # Add edge and child node
                # Syntax: Parent -- Label --> Child("NodeLabel"):::Style
                lines.append(
                    f'    {pid} -- "{child.move}" --> {cid}("{child_lbl}"):::{child_class}'
                )

                stack.append((child, cid, child_is_white))

        # Append styles
        # Gold for white, Brown for black
        lines.append("    classDef whiteTurn fill:white,color:black,stroke:#333;")
        lines.append(
            "    classDef blackTurn fill:#8b7355,color:white,stroke:#333;"
        )

        return "\n".join(lines)
    return mo, node_to_mermaid


@app.cell
def _():
    from d2_widget import Widget


    def node_to_d2(root, root_is_white: bool = True) -> str:
        """
        Convert PositionNode tree to D2 diagram syntax with labeled edges.
        Edge labels are the moves (e.g., "e4", "c5").
        Node labels are the evaluations.
        """
        lines = []
        node_counter = 0

        def get_eval_label(analysis):
            if analysis.mate is not None:
                return f"M{analysis.mate}"
            if analysis.cpl is not None:
                return f"{analysis.cpl / 100:.2f}"
            return "?"

        # Stack: (node, node_id, is_white_turn)
        root_id = f"n{node_counter}"
        node_counter += 1

        # Root node definition
        root_label = get_eval_label(root.analysis)
        root_color = "gold" if root_is_white else "saddlebrown"
        root_text_color = "black" if root_is_white else "white"

        lines.append(
            f'{root_id}: "{root_label}" {{\n'
            f"  shape: rectangle\n"
            f"  style.fill: {root_color}\n"
            f"  style.font-color: {root_text_color}\n"
            f"}}"
        )

        # Stack for DFS
        stack = [(root, root_id, root_is_white)]

        while stack:
            curr, parent_id, white_to_move = stack.pop()

            for child in curr.children:
                child_id = f"n{node_counter}"
                node_counter += 1

                child_is_white = not white_to_move
                child_label = get_eval_label(child.analysis)
                child_color = "gold" if child_is_white else "saddlebrown"
                child_text_color = "black" if child_is_white else "white"

                # Child node definition
                lines.append(
                    f'{child_id}: "{child_label}" {{\n'
                    f"  shape: rectangle\n"
                    f"  style.fill: {child_color}\n"
                    f"  style.font-color: {child_text_color}\n"
                    f"}}"
                )

                # Edge with move label
                lines.append(f'{parent_id} -> {child_id}: "{child.move}"')

                stack.append((child, child_id, child_is_white))

        return "\n".join(lines)
    return Widget, node_to_d2


@app.cell
def _():
    return


@app.cell(column=1)
def _(mo, node_to_mermaid, tree):
    # Usage
    mermaid_source = node_to_mermaid(tree)
    mo.mermaid(mermaid_source)
    return


@app.cell
def _(Widget, node_to_d2, tree):
    d2_source = node_to_d2(tree, root_is_white=True)

    # Pass D2 source and compile options
    widget_d2 = Widget(
        d2_source,
        {
            "themeID": 0,  # 0=default, 200=dark mauve, etc.
            "sketch": False,  # Set to True for hand-drawn style
            "pad": 20,
        },
    )
    widget_d2
    return


@app.cell
def _(Widget):
    Widget("""
    x: I'm a Mac {
      link: https://apple.com
    }
    y: And I'm a PC {
      tooltip: This is not a Mac
    }
    x -> y: gazoontite {
      link: https://google.com
    }
    """)
    return


@app.cell
def _(mo, node_to_dot_source, tree):
    import graphviz

    # Create graphviz object and add URLs to each node
    # We need to inject URLs into the DOT source


    def add_urls_to_dot(dot_string):
        """Inject clickable URLs into dot nodes."""
        lines = dot_string.split("\n")
        modified_lines = []

        for line in lines:
            # Find node definitions and add URL
            if "[" in line and "label=" in line and "];" in line:
                # Insert URL before the closing bracket
                line = line.replace("];", ', URL="https://google.com"];')
            modified_lines.append(line)

        return "\n".join(modified_lines)


    dot_with_urls = add_urls_to_dot(node_to_dot_source(tree, root_is_white=True))

    # Generate SVG (this works client-side in Python, no external graphviz binary needed)
    g = graphviz.Source(dot_with_urls, format="svg")
    svg_string = g.pipe(format="svg").decode("utf-8")

    # Display as interactive HTML
    mo.Html(svg_string)
    return


@app.cell
def _(mo):
    mo.mermaid(
        diagram="""
    flowchart LR
        A-->B
        B-->C
        C-->D
        click B "https://www.github.com" "This is a tooltip for a link"
        click C call callback() "Tooltip for a callback"
        click D href "https://www.github.com" "This is a tooltip for a link"

    """
    )
    return


if __name__ == "__main__":
    app.run()
