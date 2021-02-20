mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"raghulram@gmail.com\"\n\
" > ~/.streamlit/Ragulram@17.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
