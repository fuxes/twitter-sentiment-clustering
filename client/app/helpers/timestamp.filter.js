(function() {
  'use strict';

  angular
    .module('bubbleApp.helpers')
    .filter('timestamp', timestamp);

  function timestamp() {
    return function(timestamp) {
      return new Date(timestamp);
    };
  }
})();