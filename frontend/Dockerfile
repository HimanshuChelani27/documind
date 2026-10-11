FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npx ng build

FROM nginx:1.27-alpine
COPY --from=build /app/dist/frontend/browser /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Writes env.js from DOCUMIND_API_URL when the container starts, so one image works anywhere.
COPY docker-env.sh /docker-entrypoint.d/40-documind-env.sh
RUN chmod +x /docker-entrypoint.d/40-documind-env.sh
EXPOSE 80
