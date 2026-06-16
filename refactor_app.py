import os

app_path = "app.py"
with open(app_path, "r") as f:
    lines = f.readlines()

new_lines = []

for i, line in enumerate(lines):
    if line.startswith("from core.visualiser import"):
        new_lines.append(line)
        new_lines.append("from core.estimation import estimate_effect\n")
        continue
    
    if line.strip() == 'st.caption("Automated causal structure discovery")':
        new_lines.append('    st.caption("Automated causal structure & effect analysis")\n')
        continue

    if line.strip() == 'st.divider()' and "with st.sidebar:" in "".join(lines[i-5:i]):
        new_lines.append(line)
        # Add navigation right after the first divider in sidebar
        nav_code = """
    st.subheader("🧭 Navigation")
    nav_selection = st.radio(
        "Choose a workspace:",
        ["🔍 Causal Discovery", "📊 Causal Estimation", "🤖 Causal AI Agent"],
        label_visibility="collapsed"
    )

    st.divider()
"""
        new_lines.append(nav_code)
        continue

    if line.startswith("# ── Main area ─────────────────────────────────────────────────────────────────"):
        # We start the routing here
        new_lines.append(line)
        continue

    if line.strip() == 'st.title("Causal Inference Agent")':
        new_lines.append('if nav_selection == "🔍 Causal Discovery":\n')
        new_lines.append('    st.title("Causal Discovery")\n')
        continue

    if line.strip() == 'st.markdown("Discover causal structure in your data — automatically.")':
        new_lines.append('    st.markdown("Discover causal structure in your data — automatically.")\n')
        continue

    # Identify if we are inside the main area
    if i > 141: # After title and markdown
        # Indent by 4 spaces
        new_lines.append("    " + line if line.strip() else "\n")
    else:
        new_lines.append(line)

# Add the other tabs
new_lines.append("""

elif nav_selection == "📊 Causal Estimation":
    st.title("Causal Estimation")
    st.markdown("Estimate Average Treatment Effects (ATE) adjusting for confounders.")

    if uploaded_file is None:
        st.info("Upload a CSV file in the sidebar to get started.")
    else:
        with st.spinner("Loading data…"):
            df = load_data(uploaded_file)
        
        cols = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            treatment = st.selectbox("Treatment (Action)", options=cols, index=0)
        with col2:
            outcome = st.selectbox("Outcome (Result)", options=cols, index=min(1, len(cols)-1))
        
        confounders = st.multiselect(
            "Confounders (Common Causes)",
            options=[c for c in cols if c not in [treatment, outcome]],
            default=[]
        )
        
        if st.button("▶ Run Causal Analysis", type="primary"):
            with st.spinner("Running regression..."):
                try:
                    result = estimate_effect(df, treatment, outcome, confounders)
                    
                    st.success("Analysis complete.")
                    
                    st.subheader("Estimated Effect")
                    m1, m2 = st.columns(2)
                    m1.metric("Average Treatment Effect (ATE)", f"{result.ate:.4f}")
                    m2.metric("Standard Error (SE)", f"{result.se:.4f}")
                    
                    st.info(f"**Interpretation:** {result.interpretation}")
                    
                    with st.expander("Show Regression Summary", expanded=True):
                        st.text(result.summary_text)
                except Exception as e:
                    st.error(f"Estimation failed: {e}")

elif nav_selection == "🤖 Causal AI Agent":
    st.title("Causal AI Agent")
    st.markdown("Chat with the Causal AI Agent for guidance on causal inference.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about causal inference..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Mock agent response
        response = f"I am a Causal AI Agent. You asked: '{prompt}'. In a future update, I will be integrated with a full LLM backend to answer your causal inference queries!"
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

""")

with open("app_new.py", "w") as f:
    f.writelines(new_lines)

print("Created app_new.py")
