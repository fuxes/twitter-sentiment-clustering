(function() {
  'use strict';

  angular
    .module('bubbleApp', [
      'ui.router',
      'bubbleApp.helpers',
      'bubbleApp.models',
      'bubbleApp.tweetlist'
    ]);
})();