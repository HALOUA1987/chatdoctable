echo '#!/bin/bash' > wkhtmltopdf.sh
echo 'xvfb-run -a wkhtmltopdf "$@"' >> wkhtmltopdf.sh