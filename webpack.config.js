const path = require('path');

module.exports = {
  entry: './frontend/index.tsx',
  output: {
    path: path.resolve(__dirname, 'frontend/build/static/js'),
    filename: 'main.js',
  },
  module: {
    rules: [
      {
        test: /\.(ts|tsx)$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
  mode: 'production',
};
