# MBAS web app (Next.js 15)
FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM node:22-alpine
WORKDIR /web
ENV NODE_ENV=production
COPY --from=build /web/.next .next
COPY --from=build /web/node_modules node_modules
COPY --from=build /web/package.json package.json
COPY --from=build /web/next.config.ts next.config.ts
EXPOSE 3020
CMD ["npm", "start"]
